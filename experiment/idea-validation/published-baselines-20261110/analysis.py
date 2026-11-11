"""Closed-loop table of one model (stdlib, host python3.11): results/<model>/summary.json + RESULTS.zh.md.

Every arm against the existing BF16 panel-A run on the same 280 variants and seeds. Invalid slots (task completed
during settling, Cosmos only) are dropped from every arm. Success rates carry a Wilson 95% half-width in points.
"""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
from math import comb, sqrt
from pathlib import Path

SUITES = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def mcnemar(wins, losses):
    n, k = wins + losses, min(wins, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson_half(successes, n, z=1.96):
    if n == 0:
        return None
    p, d = successes / n, 1 + z * z / n
    return 100 * z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d


def reference(cfg):
    r = cfg["reference_bf16"]
    found = {}
    if r["kind"] == "jsonl":
        for path in sorted(glob.glob(r["pattern"])):
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    if e.get("arm") == r["arm"] and e.get("complete", True):
                        found[int(e["index"])] = e
    else:
        for f in Path(r["path"]).glob("*.json"):
            e = read(f)
            found[int(e.get("index", f.stem))] = e
    if len(found) != 280:
        raise RuntimeError(f"BF16 reference holds {len(found)} episodes, expected 280")
    return found


def row_for(records, valid, half=wilson_half):
    n = len(valid)
    s = sum(bool(records[i]["success"]) for i in valid)
    by_suite = {}
    for suite in SUITES:
        pool = [i for i in valid if records[i]["suite"] == suite]
        by_suite[suite] = dict(n=len(pool), successes=sum(bool(records[i]["success"]) for i in pool))
    return dict(episodes=n, successes=s, success_percent=100 * s / n if n else None, half_width_points=half(s, n),
                by_suite=by_suite, by_dimension=dict(sorted(Counter(records[i]["dimension"] for i in valid if records[i]["success"]).items())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    p = read(args.protocol)
    cfg, model = p["models"][args.model], args.model
    root = Path(p["remote_root"]) / "results" / model
    bf16 = reference(cfg)
    if sum(bool(r["success"]) for r in bf16.values()) != cfg["reference_bf16"]["successes"]:
        raise RuntimeError("BF16 reference success count differs from the protocol")
    recs = {arm: {int(f.stem): read(f) for f in (root / "loops" / arm / "episodes").glob("*.json")} for arm in p["arms"]}
    invalid = {i for r in recs.values() for i, e in r.items() if e.get("invalid")}
    rows, pairs = {"bf16": row_for(bf16, sorted(set(bf16) - invalid))}, {}
    rows["bf16"]["bpw"] = 16.0
    for arm in p["arms"]:
        r = recs[arm]
        valid = sorted(set(r) - invalid)
        quant = root / "banks" / arm / "COMPLETE.json"
        q = read(quant) if quant.exists() else {}
        rows[arm] = dict(row_for(r, valid), complete=len(valid) == 280 - len(invalid), bpw=q.get("bpw"),
                         proxy_nmse=q.get("proxy_nmse_weighted"),
                         mean_episode_seconds=sum(r[i].get("episode_seconds", 0.0) for i in valid) / len(valid) if valid else None)
        common = set(valid) & set(bf16)
        wins = sum(bool(r[i]["success"]) and not bool(bf16[i]["success"]) for i in common)
        losses = sum(bool(bf16[i]["success"]) and not bool(r[i]["success"]) for i in common)
        pairs[arm] = dict(slots=len(common), wins=wins, losses=losses, p=mcnemar(wins, losses))
    summary = dict(protocol=p["protocol"], model=model, invalid_slots=sorted(invalid), arms=rows, vs_bf16=pairs)
    save(root / "summary.json", summary)
    lines = [f"# {model}：原始配置基线，LIBERO-Plus 面板 A（280）", "",
             "| 臂 | bpw（目标层） | 成功 | 局数 | 成功率 | ±95% | 对 BF16 胜/负 | p | " + " | ".join(SUITES.values()) + " |",
             "|---|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(SUITES)]
    for arm, r in rows.items():
        v = pairs.get(arm)
        lines.append(f"| {arm}{'' if r.get('complete', True) else '（未完成）'} | "
                     f"{'—' if r.get('bpw') is None else format(r['bpw'], '.3f')} | {r['successes']} | {r['episodes']} | "
                     f"{r['success_percent']:.1f} | {r['half_width_points']:.1f} | "
                     + (f"{v['wins']} / {v['losses']} | {v['p']:.3g} | " if v else "— | — | ")
                     + " | ".join(f"{100 * s['successes'] / s['n']:.1f}" if s['n'] else "—" for s in r["by_suite"].values())
                     + " |")
    dims = sorted({d for r in rows.values() for d in r["by_dimension"]})
    lines += ["", "## 按扰动维度成功数", "", "| 臂 | " + " | ".join(dims) + " |", "|---|" + "---:|" * len(dims)]
    lines += [f"| {arm} | " + " | ".join(str(r["by_dimension"].get(d, 0)) for d in dims) + " |" for arm, r in rows.items()]
    (root / "RESULTS.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ANALYSIS " + model + " " + " ".join(f"{a}={r['successes']}/{r['episodes']}" for a, r in rows.items()), flush=True)


if __name__ == "__main__":
    main()
