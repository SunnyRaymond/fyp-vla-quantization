"""Closed-loop table (stdlib, host python3.11): results/fastwam/summary.json + RESULTS.zh.md.

Every LAQ arm against the BF16 panel-A run and against the reference arms of other experiments (the QuaRot arm, whose
weights LAQ reuses) on the same 280 variants and seeds. Success rates carry a Wilson 95% half-width in points.
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


def paired(a, b):
    common = sorted(set(a) & set(b))
    wins = sum(bool(a[i]["success"]) and not bool(b[i]["success"]) for i in common)
    losses = sum(bool(b[i]["success"]) and not bool(a[i]["success"]) for i in common)
    return dict(slots=len(common), wins=wins, losses=losses, p=mcnemar(wins, losses))


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
    alloc = read(root / "calib" / "allocation.json")["arms"]
    recs = {"bf16": bf16}
    for name, path in cfg.get("reference_arms", {}).items():
        recs[name] = {int(f.stem): read(f) for f in Path(path).glob("*.json")}
    for arm in p["arms"]:
        recs[arm] = {int(f.stem): read(f) for f in (root / "loops" / arm / "episodes").glob("*.json")}
    rows, pairs = {}, {}
    for name, r in recs.items():
        if not r:
            continue
        rows[name] = dict(row_for(r, sorted(r)), complete=len(r) == 280,
                          mean_episode_seconds=sum(e.get("episode_seconds", 0.0) for e in r.values()) / len(r))
        if name in alloc:
            acc = alloc[name]["accounting"]
            overlaps = [e["gaze_overlap_previous"] for e in r.values() if e.get("gaze_overlap_previous") is not None]
            offline = [e["gaze_overlap_offline"] for e in r.values() if e.get("gaze_overlap_offline") is not None]
            rows[name].update(a_bits=acc["a_bits"], kv_bits=acc["kv_bits"], spec=alloc[name]["spec"],
                              gaze_overlap_previous=sum(overlaps) / len(overlaps) if overlaps else None,
                              gaze_overlap_offline=sum(offline) / len(offline) if offline else None)
    for arm in p["arms"]:
        if arm in rows:
            pairs[arm] = {other: paired(recs[arm], recs[other]) for other in rows if other != arm and other not in p["arms"]}
    summary = dict(protocol=p["protocol"], model=model, arms=rows, pairs=pairs)
    save(root / "summary.json", summary)
    lines = ["# Fast-WAM：LAQ，LIBERO-Plus 面板 A（280）", "",
             "| 臂 | A bit | KV bit | 成功 | 局数 | 成功率 | ±95% | " + " | ".join(SUITES.values()) + " |",
             "|---|---:|---:|---:|---:|---:|---:|" + "---:|" * len(SUITES)]
    for name, r in rows.items():
        bits = (f"{r['a_bits']:.2f} | {r['kv_bits']:.2f}" if "a_bits" in r else "16 | 16" if name == "bf16" else "4 | 4")
        lines.append(f"| {name}{'' if r['complete'] else '（未完成）'} | {bits} | {r['successes']} | {r['episodes']} | "
                     f"{r['success_percent']:.1f} | {r['half_width_points']:.1f} | "
                     + " | ".join(f"{100 * s['successes'] / s['n']:.1f}" if s['n'] else "—" for s in r["by_suite"].values()) + " |")
    lines += ["", "## 逐局配对（同一批变体与种子，只比两臂都跑完的局）", "", "| 臂 | 对照 | 配对局数 | 胜 | 负 | p（McNemar 精确） |",
              "|---|---|---:|---:|---:|---:|"]
    for arm, d in pairs.items():
        lines += [f"| {arm} | {other} | {v['slots']} | {v['wins']} | {v['losses']} | {v['p']:.3g} |" for other, v in d.items()]
    dims = sorted({d for r in rows.values() for d in r["by_dimension"]})
    lines += ["", "## 按扰动维度成功数", "", "| 臂 | " + " | ".join(dims) + " |", "|---|" + "---:|" * len(dims)]
    lines += [f"| {name} | " + " | ".join(str(r["by_dimension"].get(d, 0)) for d in dims) + " |" for name, r in rows.items()]
    for arm in p["arms"]:
        r = rows.get(arm)
        if r and r.get("gaze_overlap_previous") is not None:
            lines += ["", f"{arm}：相邻两次 query 选中的 future 缓存 token 的 Jaccard 重合平均 {r['gaze_overlap_previous']:.3f}，"
                          f"与离线 gaze 选中集合的重合平均 {r['gaze_overlap_offline']:.3f}。"]
    (root / "RESULTS.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ANALYSIS " + model + " " + " ".join(f"{a}={r['successes']}/{r['episodes']}" for a, r in rows.items()), flush=True)


if __name__ == "__main__":
    main()
