"""CPU tests for the panel-C job: manifest builder, scheduler with per-arm totals (incl. a full 8-slot schedule
simulation), worker rows / claim / settling / patch logic, the three-panel aggregation; plus a check that panel.py's
balance rule reproduces the real panels A and B from the real pool. Run with the a16 common.py and loop_worker.py next
to these sources.
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

import analysis_c
import panel
import panel_worker as pw
import sched

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DIMS = ["objects_layout", "camera_viewpoints", "robot_initial_states", "language_instructions", "light_conditions",
        "background_textures", "sensor_noise"]
REAL = Path(__file__).resolve().parent / "protocol.json"
MEANS = dict(QUIPSHARP_W2=78.6, GPTVQ_W2=81.1, QTIP_W2=76.9, BF16=77.2, QUIPSHARP_W3=77.0, GPTVQ_W3=77.0, QTIP_W3=77.0)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fake_panels(tmp):
    """Fake 1,400 pool; panel A = first 10 of each cell, panel B = next 10 (indices 280-559)."""
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
    a_rows, b_rows = [], []
    for cell in cells.values():
        a_rows += [{**r, "index": len(a_rows) + j} for j, r in enumerate(cell[:10])]
        b_rows += [{**r, "index": 280 + len(b_rows) + j} for j, r in enumerate(cell[10:20])]
    write(tmp / "panel_a.json", dict(protocol=panel.PANEL_A_PROTOCOL, seed=7, source_manifest=dict(path=str(source)),
                                     variants=a_rows))
    return source, a_rows, b_rows


def protocol(tmp):
    p = json.loads(REAL.read_text(encoding="utf-8"))
    p["remote_root"] = str(tmp / "remote")
    p["panel_a"] = dict(p["panel_a"], manifest=str(tmp / "panel_a.json"), loops=str(tmp / "a16loops"),
                        bf16_episodes=str(tmp / "bf16_*.jsonl"))
    p["panel_b"] = dict(p["panel_b"], manifest=str(tmp / "panel_b.json"), loops=str(tmp / "pbloops"))
    p["a16"] = dict(p["a16"], root=str(tmp / "a16"))
    return p


def write_b(p, source, b_rows):
    write(p["panel_b"]["manifest"], dict(protocol=p["panel_b"]["protocol"], seed=p["panel_b"]["seed"],
                                         source_manifest=dict(path=str(source)), variants=b_rows))


def test_panel_build():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        source, a_rows, b_rows = fake_panels(tmp)
        p = protocol(tmp)
        write_b(p, source, b_rows)
        doc = panel.build(p)
        v = doc["variants"]
        used = {r["variant_id"] for r in a_rows + b_rows}
        assert len(v) == 280 and {r["variant_id"] for r in v}.isdisjoint(used)
        assert [r["index"] for r in v] == list(range(560, 840)) and [r["position"] for r in v] == list(range(280))
        assert panel.build(p) == doc
        out = tmp / "remote" / "results" / "panel" / "manifest.json"
        assert panel.write(doc, out) is True and panel.write(doc, out) is False
        assert [r["index"] for r in panel.load(p, "C")] == list(range(560, 840))
        assert len(panel.load(p, "A")) == 280 and panel.load(p, "B")[0]["index"] == 280
        try:
            panel.write(dict(doc, seed=1), out)
        except RuntimeError:
            pass
        else:
            raise AssertionError("a different manifest overwrote the frozen one")
        write_b(p, source, b_rows[:-1])
        try:
            panel.build(p)
        except RuntimeError:
            pass
        else:
            raise AssertionError("a truncated panel B was accepted")


def test_balance_reproduces_real_panels():
    """The copied rule + cell order on the real 1,400 pool gives panel A (its seed) and panel B (A removed) exactly."""
    p = json.loads(REAL.read_text(encoding="utf-8"))
    a, b = panel.read(p["panel_a"]["manifest"]), panel.read(p["panel_b"]["manifest"])
    source = panel.read(a["source_manifest"]["path"])

    def by_cell(rows):
        out = {}
        for row in rows:
            out.setdefault((row["suite"], row["dimension"]), []).append(row["variant_id"])
        return out

    picked_a = {c: [r["variant_id"] for r in rows] for c, rows in panel.select(source, set(), a["seed"], 10).items()}
    assert picked_a == by_cell(a["variants"]), "panel.select does not reproduce panel A"
    used_a = {r["variant_id"] for r in a["variants"]}
    picked_b = {c: [r["variant_id"] for r in rows]
                for c, rows in panel.select(source, used_a, p["panel_b"]["seed"], 10).items()}
    assert picked_b == by_cell(b["variants"]), "panel.select does not reproduce panel B"
    doc = panel.build(p)
    print(f"real panel C: {len(doc['variants'])} variants, "
          f"{len({r['original_task'] for r in doc['variants']})} original tasks")


def sched_protocol(tmp):
    p = protocol(tmp)
    (tmp / "remote" / "results").mkdir(parents=True)
    return p


def test_sched_rules():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        p = sched_protocol(tmp)
        assert [sched.total(p, a) for a in p["arm_order"]] == [280] * 4 + [560] * 3
        sched.init(p)
        picks = [sched.choose(p, k) for k in range(8)]
        counts = {a: picks.count(a) for a in p["arm_order"]}
        assert counts == dict(QUIPSHARP_W2=1, GPTVQ_W2=1, QTIP_W2=1, BF16=1, QUIPSHARP_W3=2, GPTVQ_W3=1,
                              QTIP_W3=1), counts
        assert sched.choose(p, 0) == "QUIPSHARP_W3"
        assert sched.pick(dict(p, arm_order=["A"]), dict(A=3), dict(A=30)) == "A"
        assert sched.pick(dict(p, arm_order=["A"]), dict(A=3), dict(A=15)) is None
        loops = tmp / "remote" / "results" / "loops" / "QTIP_W3"
        (loops / "claims" / "300").mkdir(parents=True)
        (loops / "claims" / "600").mkdir()
        write(loops / "episodes" / "600.json", {})
        assert sched.init(p)["QTIP_W3"] == 1 and sorted(os.listdir(loops / "claims")) == ["600"]
        assert sched.status(p)["QTIP_W3"] == dict(total=560, claimed=1, finished=1, active=0)
        lock = tmp / "remote" / "results" / "sched" / "lock"
        lock.mkdir()
        old = time.time() - 1000
        os.utime(lock, (old, old))
        assert sched.choose(p, 3) in p["arm_order"]


def test_schedule_simulation():
    """Event simulation of 8 slots on the real scheduler and claim directories: makespan near ideal, few reloads."""
    load, first_load, t5_done = 300.0, 560.0, 900.0
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        p = sched_protocol(tmp)
        sched.init(p)
        rng = random.Random(3)
        queue = {a: list(range(560 - 280 * (len(p["arms"][a]["panels"]) - 1), 840)) for a in MEANS}
        dur = {a: {i: MEANS[a] * rng.uniform(0.5, 1.5) for i in queue[a]} for a in MEANS}
        start = {k: 10.0 * (k - 1) for k in range(2, 8)}
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
                ready = max(now + (first_load if stints <= 8 else load), t5_done)
                heapq.heappush(events, (ready, k, "claim", arm))
            else:
                claims = tmp / "remote" / "results" / "loops" / arm / "claims"
                claims.mkdir(parents=True, exist_ok=True)
                taken = None
                for j in range(pos[k], len(queue[arm])):
                    try:
                        (claims / str(queue[arm][j])).mkdir()
                        taken = j
                        break
                    except FileExistsError:
                        continue
                if taken is None:
                    heapq.heappush(events, (now, k, "choose", None))
                    continue
                pos[k] = taken + 1
                d = dur[arm][queue[arm][taken]]
                busy += d
                heapq.heappush(events, (now + d, k, "claim", arm))
        ideal = busy / 8 + t5_done
        assert all(sched.claimed(p, a) == sched.total(p, a) for a in MEANS)
        print(f"simulated makespan {end / 3600:.2f} h, ideal {ideal / 3600:.2f} h, worker stints {stints}")
        assert end < 1.08 * ideal and stints <= 16, (end, ideal, stints)


def test_worker_rows_claim_and_order():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        source, _, b_rows = fake_panels(tmp)
        p = protocol(tmp)
        write_b(p, source, b_rows)
        panel.write(panel.build(p), tmp / "remote" / "results" / "panel" / "manifest.json")
        w3, w2 = pw.arm_rows(p, "QTIP_W3"), pw.arm_rows(p, "QTIP_W2")
        assert [r["index"] for r in w3] == list(range(280, 840)) and [r["index"] for r in w2] == list(range(560, 840))
        order = pw.task_order(w3)
        assert all(r["suite"] == "libero_10" for r in order[:140]) and order[140]["suite"] != "libero_10"
        root = tmp / "loops" / "QTIP_W3"
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
        assert sorted(got) == list(range(280, 840))


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
    row = dict(index=600, variant_id="s:1", suite="libero_goal", dimension="sensor_noise", task_id=1)
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
            pw.lw.install_weights(None, None, "QTIP_W3")
            pw.lw.install_weights(None, None, "BF16")
            assert installed == [("a16", "QTIP_W3")]
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
        p = protocol(tmp)
        mk = lambda i: dict(index=i, suite=SUITES[i % 4], dimension=DIMS[i % 7])
        write(p["panel_a"]["manifest"], dict(protocol=panel.PANEL_A_PROTOCOL, variants=[mk(i) for i in range(280)]))
        write(p["panel_b"]["manifest"], dict(protocol=p["panel_b"]["protocol"], variants=[mk(i) for i in range(280, 560)]))
        res = tmp / "remote" / "results"
        write(res / "panel" / "manifest.json", dict(protocol=p["protocol"], variants=[mk(i) for i in range(560, 840)]))
        rate = dict(BF16=0.8, QTIP_W2=0.75, GPTVQ_W2=0.72, QUIPSHARP_W2=0.7, QTIP_W3=0.78, GPTVQ_W3=0.77,
                    QUIPSHARP_W3=0.79)
        rng = random.Random(9)
        outcome = {arm: {i: rng.random() < r for i in range(840)} for arm, r in rate.items()}
        lines = [json.dumps(dict(arm="bf16", index=i, success=outcome["BF16"][i])) for i in range(280)]
        (tmp / "bf16_0.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for arm in rate:
            here = res / "loops" / arm
            if arm != "BF16":
                write(tmp / "a16" / "results" / "quantized" / arm / "COMPLETE.json", dict(bpw=2.01))
                for i in range(280):
                    write(tmp / "a16loops" / arm / "episodes" / f"{i}.json", dict(success=outcome[arm][i]))
            b_dir = here if "B" in p["arms"][arm]["panels"] else tmp / "pbloops" / arm
            for i in range(280, 840):
                rec = dict(success=outcome[arm][i], policy_call_seconds=[0.5], episode_seconds=70.0)
                if i == 700:
                    rec.update(success=False, invalid=True)
                write((b_dir if i < 560 else here) / "episodes" / f"{i}.json", rec)
            write(here / "worker_0" / "summary.json", dict(load_and_anchor_seconds=300.0))
        s = analysis_c.analyse(p)
        assert s["valid"] == dict(A=280, B=280, C=279) and s["invalid_slots"]["C"] == [700]
        assert s["invalid_consistent"] == dict(B=True, C=True)
        q = s["arms"]["QTIP_W3"]
        exp = [sum(outcome["QTIP_W3"][i] for i in range(lo, lo + 280) if i != 700) for lo in (0, 280, 560)]
        assert [q["panels"][k]["successes"] for k in "ABC"] == exp and q["total"]["episodes"] == 839
        w = sum(outcome["QTIP_W3"][i] and not outcome["BF16"][i] for i in range(840) if i != 700)
        assert q["vs_reference"]["total"]["wins"] == w and q["vs_reference"]["total"]["slots"] == 839
        assert list(s["pairs"]) == ["QUIPSHARP_W2|GPTVQ_W2", "QUIPSHARP_W2|QTIP_W2", "GPTVQ_W2|QTIP_W2",
                                    "QUIPSHARP_W3|GPTVQ_W3", "QUIPSHARP_W3|QTIP_W3", "GPTVQ_W3|QTIP_W3",
                                    "QUIPSHARP_W3|QUIPSHARP_W2", "GPTVQ_W3|GPTVQ_W2", "QTIP_W3|QTIP_W2"]
        assert q["ran_here"] == ["B", "C"] and s["arms"]["BF16"]["ran_here"] == ["C"]
        text = (res / "RESULTS.zh.md").read_text(encoding="utf-8")
        assert "合计 /839" in text and "QTIP_W3 对 QTIP_W2" in text
        # an arm missing a panel-B episode is refused
        (tmp / "pbloops" / "QTIP_W2" / "episodes" / "300.json").unlink()
        try:
            analysis_c.analyse(p)
        except RuntimeError:
            pass
        else:
            raise AssertionError("an incomplete panel was aggregated")


if __name__ == "__main__":
    tests = [test_panel_build, test_sched_rules, test_schedule_simulation, test_worker_rows_claim_and_order,
             test_settling_and_patch, test_analysis]
    if os.environ.get("FW_REAL_FILES") == "1":
        tests.insert(1, test_balance_reproduces_real_panels)
    for test in tests:
        test()
        print("ok", test.__name__, flush=True)
    print("ALL PASSED")
