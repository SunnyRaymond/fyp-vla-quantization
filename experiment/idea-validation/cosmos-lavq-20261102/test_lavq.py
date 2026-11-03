"""CPU checks before the GPU chain (cputest.pbs runs this in the Cosmos container; results/cputest/<job>/result.json).

1. every source compiles; the Cosmos runtime / endpoint / calibrate / res_sampler / LIBERO modules import
2. Capture on the pinned res_sampler with a toy denoiser behind a counted x0_fn (the endpoint's): cuts 2..4 are loop
   boundaries, five evaluations, continuation from each captured state reproduces the output, swap is differentiable
3. calibration inputs and T5 cache entries of all C80 / held-out contexts; the 280 target names fall into the 9 groups
4. QTIP K = 1 and K = 4 on small layers: decode check, storage bits, pair codes reproduce the BF16 weight bit for bit
   (layer_codes), QLayer fold is exact at zero, QTIPGrad's table gradient equals autograd through the FP32 decode
5. exchange prefixes on synthetic gains / costs: budget, donors never raised, stop rule
6. soft-bin curvature against finite differences of the reader KL; metric build / split stability on random probes
7. analysis.py on synthetic LAVQ / baseline episodes (reused BF16 panel A, an invalid slot); loop protocol generation
"""
from __future__ import annotations

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
        print(f"PASS {name} {json.dumps(value, default=str)[:600]}", flush=True)
    except Exception as exc:
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
    for name in ("cosmos_runtime", "endpoint", "calibrate", "quantization", "common", "cosmos_lavq", "qmath",
                 "ravq_qtip", "softbin_loss", "sched", "analysis", "loop_worker"):
        importlib.import_module(name)
    from cosmos_policy._src.imaginaire.modules import res_sampler
    from libero.libero import benchmark  # noqa: F401
    if not hasattr(res_sampler, "fori_loop"):
        raise RuntimeError("res_sampler has no module-level fori_loop to patch")
    os.environ.setdefault("LV_STAGE", "test")
    import lavq_worker  # noqa: F401 (reads protocol.json)
    return "ok"


def sampler_capture():
    import inspect
    import torch
    import cosmos_lavq as cl
    from cosmos_policy._src.imaginaire.modules import res_sampler
    torch.manual_seed(0)
    state = {"sampler_evaluations": 0}

    def toy(x, sigma):
        s = sigma.reshape(-1, *([1] * (x.ndim - 1))).to(x.dtype)
        mix = x.mean(2, keepdim=True)  # couples the slots, as the joint DiT does
        return 0.9 * x / (1 + s * s) + 0.1 * torch.sin(x) + 0.05 * torch.tanh(mix)

    def counted(x, sigma):
        state["sampler_evaluations"] += 1
        return toy(x, sigma)

    sampler = res_sampler.Sampler()
    sig = inspect.signature(sampler.forward)
    x_max = torch.randn(1, 16, 9, 4, 4, dtype=torch.float64) * 80
    kwargs = {}
    for name, value in (("num_steps", 5), ("sigma_min", 4.0), ("sigma_max", 80.0)):
        if name in sig.parameters:
            kwargs[name] = value
    with cl.Capture((2, 3, 4)) as cap:
        out = sampler(counted, x_max, **kwargs)
    cap.require()
    evals = state["sampler_evaluations"]
    ident = {}
    for c in (2, 3, 4):
        again = cap.continue_from(c, cap.states[c][0])
        ident[c] = float((again - out).abs().max())
    if max(ident.values()) > 1e-12:
        raise RuntimeError(f"continuation identity {ident}")
    z = cap.future(3).clone().requires_grad_(True)
    y = cap.continue_from(3, cl.with_future(cap.states[3][0], z))
    g, = torch.autograd.grad(y[:, :, 4].sum(), z)
    return dict(evaluations=evals, starts=cap.starts, identity=ident, swap_grad_norm=float(g.norm()),
                signature=list(sig.parameters))


def inputs_and_groups():
    from common import calibration_input, load_pickle, merged_cache, prep, read
    import lavq_worker as lw
    cache = load_pickle(merged_cache(P))
    missing = []
    for c, s in lw.c80() + lw.heldout():
        d = calibration_input(P, c)[1]["description"]
        if d not in cache:
            missing.append(c)
    if missing:
        raise RuntimeError(f"T5 cache misses {missing[:5]}")
    rows = read(prep(P) / "modules.json")
    groups = {}
    for r in rows:
        g, kind = lw.group_of(r["module"])
        groups[g] = groups.get(g, 0) + 1
    want = {f"b{b}_{k}": n * {"self": 4, "cross": 4, "mlp": 2}[k] for b, n in enumerate((10, 9, 9))
            for k in ("self", "cross", "mlp")}
    if groups != want:
        raise RuntimeError(f"groups {groups} != {want}")
    return dict(contexts=len(lw.c80() + lw.heldout()), groups=groups)


def qtip_codes():
    import torch
    import torch.nn.functional as F
    import qmath
    import ravq_qtip
    import lavq_worker as lw
    Q, permute, permute_inv = lw.qtip_module()
    torch.manual_seed(1)
    table = qmath.lut_values(Q.codebook(2, "cpu").tlut.to(torch.bfloat16))
    out = {}
    for K in (1, 4):
        m, n = 64, 128
        w = torch.randn(m, n) * 0.02
        x = torch.randn(512, n)
        H = x.t() @ x / x.shape[0]
        r = ravq_qtip.quantize_layer(Q, w, H, K, seed=7)
        if r["bits_total"] != Q.storage_bits(m, n, K):
            raise RuntimeError("bits")
        wq = r["dequant"].to(torch.bfloat16)
        saved = qmath.layer_codes(Q, r["codes"], r["config"], wq, table, permute_inv, "cpu")
        lin = torch.nn.Linear(n, m, bias=False)
        lin.weight.data = wq.clone()
        L = qmath.QLayer(0, "toy", lin, "g", saved, Q, permute, permute_inv, "cpu")
        L.rebuild(torch.zeros(512, 2), table)
        if not torch.equal(lin.weight.data, wq):
            raise RuntimeError("fold at zero is not exact")
        d = torch.randn(512, 2) * 1e-3
        L.rebuild(d, table)
        want = (L.decoded(table) + L.decoded(d)).to(torch.bfloat16)
        exact = float((lin.weight.data.float() - want.float()).abs().max())
        lin.weight.data = wq.clone()
        xs = torch.randn(3, 10, n).to(torch.bfloat16)
        gout = torch.randn(3, 10, m).to(torch.bfloat16)
        tab = torch.zeros(512, 2, requires_grad=True)
        y = qmath.QTIPGrad.apply(xs, tab, L)
        (y * gout).sum().backward()
        tab2 = torch.zeros(512, 2, requires_grad=True)
        D = qmath.decode(saved["pairs"], (table + tab2).to(L.dtype), m, n, permute_inv) * L.Wscale
        Wf = qmath.incoherent(Q, D, L.SU.to(L.dtype), L.SV.to(L.dtype)).float()
        (F.linear(xs.float(), Wf) * gout.float()).sum().backward()
        rel = float((tab.grad - tab2.grad).norm() / tab2.grad.norm())
        if rel > 1e-4:
            raise RuntimeError(f"K={K}: table gradient rel error {rel}")
        out[K] = dict(bpw=r["bits_total"] / (m * n), fold_vs_formula=exact, grad_rel=rel)
    return out


def exchange_rule():
    import random
    import lavq_worker as lw
    rnd = random.Random(0)
    groups = [f"g{i}" for i in range(9)]
    layers = {g: list(range(4 * i, 4 * i + 4)) for i, g in enumerate(groups)}
    sizes = {i: rnd.choice((4, 8, 16)) * 1000 for g in groups for i in layers[g]}
    bits = {i: {k: k * sizes[i] + 100 for k in (1, 2, 3, 4)} for i in sizes}
    proxy = {i: {k: 0.1 / 4 ** k * rnd.uniform(0.5, 2) for k in (1, 2, 3, 4)} for i in sizes}
    gain = {g: rnd.uniform(-1, 5) * 1e-4 for g in groups}
    cost = {g: rnd.uniform(0, 8) * 1e-4 for g in groups}
    fam = dict(K=2, down=1, up=3)
    prefixes, donors, log, dens = lw.exchanges(fam, groups, gain, cost, layers, bits, proxy, sizes, 4)
    base = sum(bits[i][2] for i in sizes)
    for j, K in enumerate(prefixes):
        total = sum(bits[i][K[i]] for i in K)
        if total > base:
            raise RuntimeError(f"prefix {j} over budget")
        for d in donors[:j]:
            if any(K[i] != 1 for i in layers[d]):
                raise RuntimeError("donor not lowered")
        raised = {g for g in groups if any(K[i] == 3 for i in layers[g])}
        if raised & set(donors[:j]):
            raise RuntimeError("a donor was raised")
    return dict(prefixes=len(prefixes), donors=donors, log=log[-1] if log else None)


def curvature_and_metric():
    import torch
    import cosmos_lavq as cl
    import qmath
    torch.manual_seed(2)
    lo, hi = -torch.ones(7), torch.ones(7)
    stats = dict(actions_min=lo, actions_max=hi, mean=torch.zeros(6), scale=torch.full((6,), 0.3))
    t = torch.rand(1, 16, 7) * 1.2 - 0.6
    h = cl.kl_curvature(t, stats, P["reader"])
    eps = 1e-3
    worst = 0.0
    for idx in ((0, 0, 0), (0, 5, 3), (0, 15, 5)):
        e = torch.zeros_like(t)
        e[idx] = eps
        k_p = float(cl.reader_kl(t + e, t, stats, P["reader"]))
        k_m = float(cl.reader_kl(t - e, t, stats, P["reader"]))
        fd = (k_p + k_m) / eps ** 2  # K(t) = 0
        worst = max(worst, abs(fd - float(h[idx])) / max(float(h[idx]), 1e-12))
    if worst > 1e-2:
        raise RuntimeError(f"curvature vs finite differences {worst}")
    vectors = [torch.randn(4, 200) * (1 + i % 3) for i in range(20)]
    saved, st, diag = qmath.build_metric(vectors, list(range(20)), 8, 0.1)
    M = qmath.Metric(saved, "cpu")
    v = torch.randn(200)
    zero = float(M.nmse(v, v))
    stab = qmath.split_stability(vectors, list(range(20)), 8)
    return dict(curvature_fd_rel=worst, effective_rank=st["effective_rank"], zero=zero, stability=stab)


def loop_analysis():
    import analysis
    from common import save
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = tmp / "base"
        old = tmp / "old"
        bp = dict(reuse=dict(BF16_A=dict(episodes=str(old))))
        (base).mkdir()
        (base / "protocol.json").write_text(json.dumps(bp))
        suites = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
        for i in range(840):
            rec = dict(success=(i * 7) % 5 != 0, suite=suites[i % 4], dimension=f"d{i % 7}", invalid=i == 300)
            for arm in ("BF16", "QTIP_W2", "QUIPSHARP_W2", "GPTVQ_W2"):
                if arm == "BF16" and i < 280:
                    save(old / f"{i:03d}.json", rec)
                else:
                    save(base / f"results/loops/{arm}/episodes/{i}.json", dict(rec, success=rec["success"] ^ (i % 11 == 0)))
            save(tmp / f"new/results/loops/LAVQ_W2/episodes/{i}.json", dict(rec, success=rec["success"] or i % 13 == 0))
        lp = dict(protocol="t", remote_root=str(tmp / "new"), baseline_root=str(base), arm_order=["LAVQ_W2"],
                  arms={"LAVQ_W2": {"panels": ["A", "B", "C"]}})
        (tmp / "lp.json").write_text(json.dumps(lp))
        sys.argv = ["analysis.py", "--protocol", str(tmp / "lp.json")]
        analysis.main()
        s = json.loads((tmp / "new/results/loop_summary.json").read_text())
        if s["invalid_slots"] != [300] or s["arms"]["LAVQ_W2"]["episodes"] != 839 or s["arms"]["BF16"]["episodes"] != 839:
            raise RuntimeError(f"analysis counts wrong {s['invalid_slots']} {s['arms']['LAVQ_W2']['episodes']}")
        return dict(pairs=list(s["pairs"]), lavq=s["arms"]["LAVQ_W2"]["successes"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    check("compile", compile_all)
    check("imports", imports)
    check("sampler_capture", sampler_capture)
    check("inputs_and_groups", inputs_and_groups)
    check("exchange_rule", exchange_rule)
    check("curvature_and_metric", curvature_and_metric)
    check("loop_analysis", loop_analysis)
    check("qtip_codes", qtip_codes)
    ok = all(v["ok"] for v in REPORT.values())
    (OUT / "result.json").write_text(json.dumps(dict(ok=ok, checks=REPORT), indent=2, default=str) + "\n")
    print("ALL_PASS" if ok else "SOME_FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
