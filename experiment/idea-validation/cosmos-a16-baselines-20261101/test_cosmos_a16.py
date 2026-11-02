"""CPU checks before the GPU chain (cputest.pbs runs this in the Cosmos container; results/cputest/<job>/).

1. every source compiles; the Cosmos runtime, endpoint, quantization, T5 encoder and LIBERO modules import
2. panels A/B/C load (280 each, slot indices 0-839); B/C instructions vs the base+extra T5 cache (count missing)
3. every method x bit width on two small layers: dequant shape, bpw ~ bits, codes present and decodable to the dequant
4. shard split of the 280 Cosmos shapes; scheduler pick/choose/status and BF16 panel A reuse rule on a temp root
5. loop_worker claim order; analysis on synthetic episodes (incl. the reused BF16 panel A and an invalid slot)
6. loop_worker T5 staging on the real merged cache for all 840 slots
(cputest.pbs also checks that loop_worker.gpu_lock excludes across two separate containers.)
"""
from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import py_compile
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("TEST_OUT", HERE / "test_out"))
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
REPORT = {}


def check(name, fn):
    start = time.monotonic()
    try:
        value = fn()
        REPORT[name] = dict(ok=True, seconds=round(time.monotonic() - start, 1), value=value)
        print(f"PASS {name} {json.dumps(value, default=str)[:400]}", flush=True)
    except Exception as exc:  # report every failing check, not just the first
        import traceback
        traceback.print_exc()
        REPORT[name] = dict(ok=False, seconds=round(time.monotonic() - start, 1), error=f"{type(exc).__name__}: {exc}")
        print(f"FAIL {name} {type(exc).__name__}: {exc}", flush=True)


def compile_all():
    files = [f for f in HERE.rglob("*.py") if "__pycache__" not in f.parts]
    for f in files:
        py_compile.compile(str(f), doraise=True)
    return len(files)


def imports():
    for name in ("cosmos_runtime", "endpoint", "quantization", "common", "sched", "analysis", "loop_worker",
                 "prep_t5", "prep_hess", "quant_worker"):
        importlib.import_module(name)
    from cosmos_policy._src.predict2.inference.get_t5_emb import CosmosT5TextEncoder  # noqa: F401
    from libero.libero import benchmark  # noqa: F401
    return "ok"


def panels_and_t5():
    from common import base_extra_cache, panel_rows
    import prep_t5
    counts = {name: len(panel_rows(P, name)) for name in ("A", "B", "C")}
    texts = prep_t5.instructions(P)
    cache, base, extra = base_extra_cache(P)
    missing = {name: sum(1 for t, n in texts.items() if n == name and t not in cache) for name in ("A", "B", "C")}
    if missing["A"]:
        raise RuntimeError(f"panel A instructions missing from the cache: {missing['A']}")
    return dict(rows=counts, instructions=len(texts), base=len(base), extra=len(extra), merged=len(cache),
                missing_by_first_panel=missing)


def methods():
    import torch
    torch.manual_seed(0)
    found = {}
    for method in ("gptq", "gptvq", "qtip", "quipsharp"):
        sys.path.insert(0, str(HERE / "methods" / method))
        sys.modules.pop("quant", None)
        q = importlib.import_module("quant")
        for bits in (2, 3):
            for m, n in ((256, 512), (512, 256)):
                x = torch.randn(4096, n) * torch.linspace(0.2, 2.0, n)
                h = x.t() @ x / x.shape[0]
                w = torch.randn(m, n) * 0.02
                r = q.quantize_layer(w, h, bits, seed=7)
                dq = r["dequant"].float()
                if tuple(dq.shape) != (m, n) or not torch.isfinite(dq).all():
                    raise RuntimeError(f"{method} W{bits} {m}x{n}: bad dequant")
                bpw = r["bits_total"] / (m * n)
                if not bits <= bpw < bits + 0.35:
                    raise RuntimeError(f"{method} W{bits} {m}x{n}: bpw {bpw}")
                c = r["codes"]
                if method == "gptq":
                    rec = c["scale"][:, None] * (c["q"].float() - c["zero"][:, None].float())
                elif method == "gptvq":
                    rec = q.decode(c)[:m, :n]
                elif method == "qtip":
                    cb = q.codebook(c["K"], torch.device("cpu"))
                    m_p, n_p = c["padded_shape"]
                    dec = q.decode_packed(c["trellis"], c["tlut"], cb, c["K"], m_p, n_p, c["kernel_layout"]).float()
                    rec = q.inverse_incoherence(dec * c["Wscale"], c["SU"].float(), c["SV"].float())[:m, :n]
                else:
                    m_p, n_p = c["padded_shape"]
                    with q.strict_fp32():
                        rec = q.decode_stored(c["attr"], bits, m_p, n_p, q.make_args(bits, n_p))[:m, :n]
                err = float((rec.float() - dq).abs().max() / dq.abs().max())
                if err > (1e-2 if method == "qtip" else 1e-6):
                    raise RuntimeError(f"{method} W{bits} {m}x{n}: codes decode to a different weight ({err:.3g})")
                d = dq - w
                proxy = float(((d @ h) * d).sum() / ((w @ h) * w).sum())
                found[f"{method}_W{bits}_{m}x{n}"] = dict(bpw=round(bpw, 4), proxy=round(proxy, 5),
                                                          decode_rel_err=err)
        sys.path.remove(str(HERE / "methods" / method))
    return found


def cosmos_shapes():
    rows, i = [], 0
    for block in range(28):
        for name, shape in (("self_attn.q_proj", (2048, 2048)), ("self_attn.k_proj", (2048, 2048)),
                            ("self_attn.v_proj", (2048, 2048)), ("self_attn.output_proj", (2048, 2048)),
                            ("cross_attn.q_proj", (2048, 2048)), ("cross_attn.k_proj", (2048, 1024)),
                            ("cross_attn.v_proj", (2048, 1024)), ("cross_attn.output_proj", (2048, 2048)),
                            ("mlp.layer1", (8192, 2048)), ("mlp.layer2", (2048, 8192))):
            rows.append(dict(index=i, module=f"net.blocks.{block}.{name}", shape=list(shape)))
            i += 1
    return rows


def shards():
    from quant_worker import shard_layers
    rows = cosmos_shapes()
    owner = shard_layers(rows, 2)
    load = [sum(r["shape"][0] * r["shape"][1] + r["shape"][1] ** 2 for r in rows if owner[r["index"]] == s)
            for s in (0, 1)]
    if abs(load[0] - load[1]) / sum(load) > 0.02 or len(owner) != 280:
        raise RuntimeError(f"unbalanced shards {load}")
    return dict(loads=load)


def temp_protocol(root):
    p = copy.deepcopy(P)
    p["remote_root"] = str(root)
    return p


def scheduler():
    import sched
    from common import bf16_panels, save
    with tempfile.TemporaryDirectory() as tmp:
        p = temp_protocol(tmp)
        save(Path(tmp) / "results/prep/reuse_check.json", dict(passed=True))
        if bf16_panels(p) != ["B", "C"] or sched.total(p, "BF16") != 560 or sched.total(p, "GPTQ_W2") != 280:
            raise RuntimeError("BF16 reuse rule / totals wrong with a passing replay")
        sched.init(p)
        picks = [sched.choose(p, k) for k in range(20)]
        for k in range(30):  # claim most of QTIP_W2: joining it must stop below join_min per worker
            (Path(tmp) / f"results/loops/QTIP_W2/claims/{k}").mkdir(parents=True)
        status = sched.status(p)
        save(Path(tmp) / "results/prep/reuse_check.json", dict(passed=False))
        if bf16_panels(p) != ["A", "B", "C"] or sched.total(p, "BF16") != 840:
            raise RuntimeError("BF16 must run panel A when the replay fails")
        # 20 slots: the six 840-episode arms 3 each, BF16 (560) 2, GPTQ (280) none until a slot frees up (ties go to
        # the earlier arm), so every worker starts with 280 unclaimed episodes.
        per_arm = {arm: picks.count(arm) for arm in P["arm_order"]}
        if [per_arm[a] for a in P["arm_order"]] != [2, 3, 3, 3, 3, 3, 3, 0, 0]:
            raise RuntimeError(f"20 slots should balance unclaimed episodes per worker: {per_arm}")
        return dict(picks=picks, qtip_w2=status["QTIP_W2"])


def t5_staging():
    """Loop-worker T5 path on the real merged cache: the one-entry anchor file, then every panel slot's language
    (resolved like CosmosRuntime.episode) staged beside it, bitwise equal to the merged entry."""
    import types
    import torch
    import loop_worker
    from common import load_pickle, panel_rows
    from cosmos_runtime import CosmosRuntime
    from libero.libero import benchmark
    with tempfile.TemporaryDirectory() as tmp:
        full, keep, small = loop_worker.t5_cache(P, Path(tmp))
        rt = types.SimpleNamespace(suites={}, benchmark=benchmark,
                                   cosmos_utils=types.SimpleNamespace(t5_text_embeddings_cache=load_pickle(small)))
        rt._suite = types.MethodType(CosmosRuntime._suite, rt)
        cache = rt.cosmos_utils.t5_text_embeddings_cache
        if list(cache) != [keep] or not torch.equal(cache[keep], full[keep]):
            raise RuntimeError("one-entry anchor cache differs from the merged cache")
        for name in ("A", "B", "C"):
            for row in panel_rows(P, name):
                language = loop_worker.stage_language(rt, full, keep, row)
                if set(cache) != {keep, language} or not torch.equal(cache[language], full[language]):
                    raise RuntimeError(f"T5 staging wrong for slot {row['index']}")
    return dict(merged=len(full), anchor=keep)


def claims_and_analysis():
    import analysis
    import loop_worker
    from common import panel_rows, save
    with tempfile.TemporaryDirectory() as tmp:
        p = temp_protocol(tmp)
        root = Path(tmp) / "results/loops/QTIP_W2"
        order = loop_worker.task_order(loop_worker.arm_rows(p, "QTIP_W2"))
        if len(order) != 840 or order[0]["suite"] != "libero_10":
            raise RuntimeError("claim order must cover 840 slots, libero_10 first")
        a, k = loop_worker.claim(root, order, 0, 0)
        b, _ = loop_worker.claim(root, order, 0, 1)
        if a["index"] == b["index"]:
            raise RuntimeError("two workers claimed the same slot")
        old = Path(tmp) / "old_bf16"
        p["reuse"]["BF16_A"]["episodes"] = str(old)
        save(Path(tmp) / "results/prep/reuse_check.json", dict(passed=True))
        rows = {n: panel_rows(p, n) for n in ("A", "B", "C")}
        for arm in p["arm_order"]:
            for name in ("A", "B", "C") if arm != "GPTQ_W3" and arm != "GPTQ_W2" else ("A",):
                for r in rows[name]:
                    i = int(r["index"])
                    rec = dict(index=i, suite=r["suite"], dimension=r["dimension"], success=(i + len(arm)) % 5 != 0,
                               invalid=i == 300, episode_seconds=80.0)
                    if arm == "BF16" and name == "A":
                        save(old / f"{i:03d}.json", rec)
                    else:
                        save(Path(tmp) / f"results/loops/{arm}/episodes/{i}.json", rec)
        sys.argv = ["analysis.py", "--protocol", str(Path(tmp) / "protocol.json")]
        (Path(tmp) / "protocol.json").write_text(json.dumps(p), encoding="utf-8")
        analysis.main()
        s = json.loads((Path(tmp) / "results/summary.json").read_text())
        if s["arms"]["BF16"]["episodes"] != 839 or not s["arms"]["BF16"]["complete"] or s["invalid_slots"] != [300]:
            raise RuntimeError(f"analysis counts wrong: {s['arms']['BF16']}")
        return dict(bf16=s["arms"]["BF16"]["successes"], qtip_vs_bf16=s["vs_bf16"]["QTIP_W2"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    check("compile", compile_all)
    check("imports", imports)
    check("panels_and_t5", panels_and_t5)
    check("shards", shards)
    check("scheduler", scheduler)
    check("claims_and_analysis", claims_and_analysis)
    check("t5_staging", t5_staging)
    check("methods", methods)
    ok = all(v["ok"] for v in REPORT.values())
    (OUT / "result.json").write_text(json.dumps(dict(ok=ok, checks=REPORT), indent=2, default=str) + "\n")
    print("ALL_PASS" if ok else "SOME_FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
