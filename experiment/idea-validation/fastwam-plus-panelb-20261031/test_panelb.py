"""CPU tests for panel B: manifest builder, scheduler (incl. a full schedule simulation), worker claim / settling /
patch logic, aggregation; plus a check that panel.py's balance rule reproduces the real panel A from the real pool.
Run with ARTIFACTS on sys.path (a16 common.py and loop_worker.py next to these sources).
"""
from __future__ import annotations

import heapq
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
import time
import types

# loop_worker.episode imports these before touching the runtime; the settling test never reaches them.
for _name in ("numpy", "torch", "runner"):
    try:
        if _name == "runner":
            raise ImportError
        __import__(_name)
    except ImportError:
        sys.modules[_name] = types.ModuleType(_name)

import analysis_b
import panel
import panel_worker as pw
import sched

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DIMS = ["objects_layout", "camera_viewpoints", "robot_initial_states", "language_instructions", "light_conditions",
        "background_textures", "sensor_noise"]
REAL = Path(__file__).resolve().parent / "protocol.json"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fake_pool(tmp):
    rng = random.Random(1)
    rows = []
    for s in SUITES:
        for d in DIMS:
            for k in range(50):
                rows.append(dict(suite=s, dimension=d, task_id=1000 + len(rows), task_name=f"t{len(rows)}",
                                 original_task=f"{s}_task{rng.randrange(10)}", subtype=f"sub{rng.randrange(4)}",
                                 difficulty=rng.randrange(1, 6), variant_id=f"{s}:{1000 + len(rows)}", state_id=0))
    source = tmp / "pool.json"
    write(source, dict(protocol=panel.SOURCE_PROTOCOL, seed=5, source=dict(repository="x"), variants=rows))
    cells = {}
    for r in rows:
        cells.setdefault((r["suite"], r["dimension"]), []).append(r)
    a_rows = []
    for (s, d), cell in cells.items():
        for r in cell[:10]:
            a_rows.append({**r, "index": len(a_rows)})
    a = tmp / "panel_a.json"
    write(a, dict(protocol=panel.PANEL_A_PROTOCOL, seed=7, source_manifest=dict(path=str(source)), variants=a_rows))
    return source, a, a_rows


def protocol(tmp, panel_a_path="unused"):
    p = json.loads(REAL.read_text(encoding="utf-8"))
    p["remote_root"] = str(tmp / "remote")
    p["panel_a"] = dict(p["panel_a"], manifest=str(panel_a_path))
    return p


def test_panel_build():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        _, a_path, a_rows = fake_pool(tmp)
        p = protocol(tmp, a_path)
        doc = panel.build(p)
        v = doc["variants"]
        assert len(v) == 280 and {r["variant_id"] for r in v}.isdisjoint({r["variant_id"] for r in a_rows})
        assert [r["index"] for r in v] == list(range(280, 560)) and [r["position"] for r in v] == list(range(280))
        assert panel.build(p) == doc
        out = tmp / "m.json"
        assert panel.write(doc, out) is True and panel.write(doc, out) is False
        other = dict(doc, seed=1)
        try:
            panel.write(other, out)
        except RuntimeError:
            pass
        else:
            raise AssertionError("a different manifest overwrote the frozen one")


def test_balance_reproduces_real_panel_a():
    """The copied rule + cell order, run on the real 1,400 pool with panel A's seed, must give panel A exactly."""
    p = json.loads(REAL.read_text(encoding="utf-8"))
    a = panel.read(p["panel_a"]["manifest"])
    source = panel.read(a["source_manifest"]["path"])
    cells = {}
    for row in source["variants"]:
        cells.setdefault((row["suite"], row["dimension"]), []).append(row)
    picked = {cell: [r["variant_id"] for r in panel.balanced_sample(rows, a["seed"] + i, 10)]
              for i, (cell, rows) in enumerate(cells.items())}
    actual = {}
    for row in a["variants"]:
        actual.setdefault((row["suite"], row["dimension"]), []).append(row["variant_id"])
    assert picked == actual, "panel.balanced_sample does not reproduce panel A"
    doc = panel.build(p)
    assert len(doc["variants"]) == 280
    print(f"real panel B: {len(doc['variants'])} variants, "
          f"{len({r['original_task'] for r in doc['variants']})} original tasks")


def sched_protocol(tmp):
    p = protocol(tmp)
    (tmp / "remote" / "results").mkdir(parents=True)
    return p


def test_sched_rules():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        p = sched_protocol(tmp)
        sched.init(p)
        picks = [sched.choose(p, k) for k in range(10)]
        counts = {a: picks.count(a) for a in p["arm_order"]}
        assert counts == dict(QUIPSHARP_W2=3, GPTVQ_W2=3, QTIP_W2=2, BF16=2), counts
        # re-choosing a slot first removes its own entry
        assert sched.choose(p, 0) == "QUIPSHARP_W2"
        # join rule: 30 unclaimed left over 3 workers (10 each) admits a 4th; 15 over 3 (5 each) does not
        assert sched.pick(dict(p, arm_order=["A"]), dict(A=3), dict(A=30)) == "A"
        assert sched.pick(dict(p, arm_order=["A"]), dict(A=3), dict(A=15)) is None
        assert sched.pick(dict(p, arm_order=["A"]), {}, dict(A=1)) == "A"
        assert sched.pick(dict(p, arm_order=["A", "B"]), dict(A=2, B=1), dict(A=100, B=40)) == "A"
        # init drops a claim without an episode and keeps a finished one
        loops = tmp / "remote" / "results" / "loops" / "BF16"
        (loops / "claims" / "300").mkdir(parents=True)
        (loops / "claims" / "301").mkdir()
        write(loops / "episodes" / "301.json", {})
        assert sched.init(p)["BF16"] == 1 and sorted(os.listdir(loops / "claims")) == ["301"]
        assert sched.slots(p) == {}
        # a stale lock is broken
        lock = tmp / "remote" / "results" / "sched" / "lock"
        lock.mkdir()
        old = time.time() - 1000
        os.utime(lock, (old, old))
        assert sched.choose(p, 3) in p["arm_order"]


def test_schedule_simulation():
    """Event simulation of 10 slots on the real scheduler and claim directories: makespan near ideal, few reloads."""
    load, t5_done = 330.0, 420.0
    means = dict(QUIPSHARP_W2=80.7, GPTVQ_W2=79.2, QTIP_W2=78.4, BF16=75.0)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        p = sched_protocol(tmp)
        sched.init(p)
        rng = random.Random(3)
        dur = {a: [means[a] * rng.uniform(0.5, 1.5) for _ in range(280)] for a in means}
        start = {k: 10.0 * (k - 1) for k in range(2, 10)}
        start.update({0: t5_done, 1: t5_done})
        events = [(s, k, "choose", None) for k, s in start.items()]
        heapq.heapify(events)
        pos, stints, busy, end = {}, 0, 0.0, 0.0
        while events:
            now, k, kind, arm = heapq.heappop(events)
            if kind == "choose":
                arm = sched.choose(p, k)
                if arm is None:
                    end = max(end, now)
                    continue
                stints += 1
                pos[k] = 0
                heapq.heappush(events, (max(now + load, t5_done), k, "claim", arm))
            else:
                claims = tmp / "remote" / "results" / "loops" / arm / "claims"
                claims.mkdir(parents=True, exist_ok=True)
                taken = None
                for i in range(pos[k], 280):
                    try:
                        (claims / str(i)).mkdir()
                        taken = i
                        break
                    except FileExistsError:
                        continue
                if taken is None:
                    heapq.heappush(events, (now, k, "choose", None))
                    continue
                pos[k] = taken + 1
                busy += dur[arm][taken]
                heapq.heappush(events, (now + dur[arm][taken], k, "claim", arm))
        ideal = busy / 10 + t5_done
        assert all(sched.claimed(p, a) == 280 for a in means)
        print(f"simulated makespan {end / 3600:.2f} h, ideal {ideal / 3600:.2f} h, worker stints {stints}")
        assert end < 1.12 * ideal and stints <= 18, (end, ideal, stints)


def test_worker_claim_and_order():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        rows = [dict(index=280 + i, suite="libero_10" if i % 4 == 3 else "libero_goal") for i in range(280)]
        order = pw.task_order(rows)
        assert all(r["suite"] == "libero_10" for r in order[:70]) and order[0]["index"] == 283
        got, starts = [], {0: 0, 1: 0}
        while True:
            progressed = False
            for w in (0, 1):
                row, starts[w] = pw.claim(root, order, starts[w], w)
                if row is not None:
                    got.append(row["index"])
                    progressed = True
            if not progressed:
                break
        assert sorted(got) == list(range(280, 560)) and len(set(got)) == 280


class FakeEnv:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeRuntime:
    def __init__(self):
        self.env = None

    def task(self, row, index):
        self.env = FakeEnv()
        raise RuntimeError("Task completed during settling; invalid evaluation slot")


def test_settling_and_patch():
    os.environ.setdefault("PBS_JOBID", "1.test")
    runtime = FakeRuntime()
    row = dict(index=300, variant_id="s:1", suite="libero_goal", dimension="sensor_noise", task_id=1)
    record = pw.run_episode(runtime, row, "BF16")
    assert record["invalid"] and not record["success"] and runtime.env is None
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        p = protocol(tmp)
        p["schedule"] = dict(p["schedule"], t5_wait_seconds=30)
        installed, saved = [], (pw.lw.install_weights, pw.lw.bf16_runtime)
        pw.lw.install_weights = lambda pa, runtime, arm: installed.append((pa["tag"], arm))
        pw.lw.bf16_runtime = lambda out, text_encoder=True: "runtime"
        try:
            waited = pw.patch(p, dict(tag="a16"))
            pw.lw.install_weights(None, None, "QTIP_W2")
            pw.lw.install_weights(None, None, "BF16")
            assert installed == [("a16", "QTIP_W2")]
            prep = tmp / "remote" / "results" / "prep"
            prep.mkdir(parents=True)
            threading.Timer(1.0, lambda: write(prep / "T5_COMPLETE.json", {})).start()
            assert pw.lw.bf16_runtime("o", text_encoder=False) == "runtime" and waited["t5_wait_seconds"] >= 0.9
            (prep / "T5_COMPLETE.json").unlink()
            write(prep / "T5_FAILED.json", {})
            try:
                pw.lw.bf16_runtime("o", text_encoder=False)
            except RuntimeError:
                pass
            else:
                raise AssertionError("worker kept waiting after T5 failed")
        finally:
            pw.lw.install_weights, pw.lw.bf16_runtime = saved


def test_analysis():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        p = protocol(tmp, tmp / "a.json")
        rows_a = [dict(index=i, suite=SUITES[i % 4], dimension=DIMS[i % 7]) for i in range(280)]
        rows_b = [dict(index=280 + i, position=i, suite=SUITES[i % 4], dimension=DIMS[i % 7]) for i in range(280)]
        write(tmp / "a.json", dict(variants=rows_a))
        res = tmp / "remote" / "results"
        write(res / "panel" / "manifest.json", dict(protocol=p["protocol"], variants=rows_b))
        p["panel_a"]["loops"] = str(tmp / "a16loops")
        p["panel_a"]["bf16_episodes"] = str(tmp / "bf16_*.jsonl")
        p["a16"] = dict(p["a16"], root=str(tmp / "a16"))
        rate = dict(BF16=0.8, QTIP_W2=0.75, GPTVQ_W2=0.72, QUIPSHARP_W2=0.7)
        rng = random.Random(9)
        outcome = {arm: {i: rng.random() < r for i in range(560)} for arm, r in rate.items()}
        lines = [json.dumps(dict(arm="bf16", index=i, success=outcome["BF16"][i])) for i in range(280)]
        (tmp / "bf16_0.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for arm in rate:
            if arm != "BF16":
                write(tmp / "a16" / "results" / "quantized" / arm / "COMPLETE.json", dict(bpw=2.01))
                for i in range(280):
                    write(tmp / "a16loops" / arm / "episodes" / f"{i}.json", dict(success=outcome[arm][i]))
            for i in range(280, 560):
                rec = dict(success=outcome[arm][i], policy_call_seconds=[0.5], episode_seconds=70.0)
                if i == 300:
                    rec.update(success=False, invalid=True)
                write(res / "loops" / arm / "episodes" / f"{i}.json", rec)
            write(res / "loops" / arm / "worker_0" / "summary.json", dict(load_and_anchor_seconds=330.0))
        s = analysis_b.analyse(p)
        assert s["panel_b_valid"] == 279 and s["invalid_slots"] == [300] and s["invalid_consistent"]
        q = s["arms"]["QTIP_W2"]
        exp_a = sum(outcome["QTIP_W2"][i] for i in range(280))
        exp_b = sum(outcome["QTIP_W2"][i] for i in range(280, 560) if i != 300)
        assert (q["panel_a"]["successes"], q["panel_b"]["successes"], q["total"]["episodes"]) == (exp_a, exp_b, 559)
        v = q["vs_reference"]["total"]
        w = sum(outcome["QTIP_W2"][i] and not outcome["BF16"][i] for i in range(560) if i != 300)
        assert v["wins"] == w and v["slots"] == 559
        assert set(s["pairs"]) == {"QUIPSHARP_W2|GPTVQ_W2", "QUIPSHARP_W2|QTIP_W2", "GPTVQ_W2|QTIP_W2"}
        assert abs(analysis_b.mcnemar(13, 8) - 0.3833) < 1e-3 and analysis_b.mcnemar(0, 0) == 1.0
        text = (res / "RESULTS.zh.md").read_text(encoding="utf-8")
        assert "合计 /559" in text and "QTIP_W2 对 GPTVQ_W2" not in text and "GPTVQ_W2 对 QTIP_W2" in text


if __name__ == "__main__":
    tests = [test_panel_build, test_sched_rules, test_schedule_simulation, test_worker_claim_and_order,
             test_settling_and_patch, test_analysis]
    if os.environ.get("FW_REAL_FILES") == "1":
        tests.insert(1, test_balance_reproduces_real_panel_a)
    for test in tests:
        test()
        print("ok", test.__name__, flush=True)
    print("ALL PASSED")
