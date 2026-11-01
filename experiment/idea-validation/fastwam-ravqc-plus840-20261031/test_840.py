"""CPU tests (host python3.11, stdlib) for the 840-episode RAVQ loop: the gate (held-out mixed loss vs the uncalibrated
base; the uncalibrated RAVQU arms always run), the install anchor reference, panel loading, the scheduler's totals (840
per calibrated arm, 560 per RAVQU arm) and slot choices, and the A + B + C aggregation with RAVQU's panel A from its own
loop and partial references."""
from __future__ import annotations

import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis_840 as an  # noqa: E402
import gate  # noqa: E402
import panel  # noqa: E402
import sched  # noqa: E402

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
DIMS = ["objects_layout", "camera_viewpoints", "robot_initial_states", "language_instructions", "light_conditions",
        "background_textures", "sensor_noise"]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fake(tmp):
    p = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    p["remote_root"] = str(tmp / "job")
    p["ravqc"]["root"] = str(tmp / "guided")
    for name, key, proto in (("A", "panel_a", panel.PANEL_A_PROTOCOL), ("B", "panel_b", p["panel_b"]["protocol"]),
                             ("C", "panel_c", p["panel_c"]["protocol"])):
        off = panel.PANELS.index(name) * panel.PANEL
        rows = [dict(index=off + k, suite=SUITES[(k // 70) % 4], dimension=DIMS[(k // 10) % 7], variant_id=f"{name}{k}",
                     task_id=k) for k in range(panel.PANEL)]
        write(tmp / f"manifest_{name}.json", dict(protocol=proto, variants=rows))
        p[key]["manifest"] = str(tmp / f"manifest_{name}.json")
    p["panel_a"]["loops"] = str(tmp / "a16_loops")
    p["panel_a"]["ravq_loops"] = str(tmp / "guided" / "results" / "loops")
    p["panel_a"]["bf16_episodes"] = str(tmp / "rot" / "worker_*" / "episodes.jsonl")
    p["panel_b"]["loops"] = str(tmp / "panelb_loops")
    p["panel_c"]["loops"] = str(tmp / "panelc_loops")
    return p


def test_gate():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        p = fake(tmp)
        g = tmp / "guided" / "results"
        for arm, mixed, base_mixed in (("RAVQC_W2", 0.0019, 0.00203), ("RAVQC_W3", 0.00062, 0.00060)):
            write(g / "ravqc" / arm / "arm.json", dict(tuned=True))
            o = dict(mixed_loss=mixed, future_latent_nmse=0.03, first10motor_rmse=0.02, first10gripper_rmse=0.04)
            write(g / "ravqc" / "heldout" / f"{arm}.json", dict(overall=o))
            write(g / "heldout" / f"{p['arms'][arm]['base']}.json", dict(overall=dict(o, mixed_loss=base_mixed)))
        rows = gate.decide(p, tmp / "guided")
        assert rows["RAVQC_W2"]["run"] and not rows["RAVQC_W3"]["run"], rows
        assert rows["RAVQU_W2"]["run"] and rows["RAVQU_W3"]["run"], rows
        write(g / "ravqc" / "RAVQC_W2" / "arm.json", dict(tuned=False))
        assert not gate.decide(p, tmp / "guided")["RAVQC_W2"]["run"]


def test_gate_cli():
    """gate.py as run.pbs calls it, and run.pbs's own parse of its output (26073907 took the GATE json line for arm
    names when no arm passed): the arms come from the RUN line only, and no passing arm gives an empty list."""
    parse = r'''read -r -a ARMS <<< "$(sed -n 's/^RUN *//p' <<< "$GATE_OUT")"; echo "${#ARMS[@]} ${ARMS[*]}"'''
    for w2_mixed, want in ((0.0019, ["RAVQU_W2", "RAVQU_W3", "RAVQC_W2"]), (0.0021, ["RAVQU_W2", "RAVQU_W3"])):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            p = fake(tmp)
            g = tmp / "guided" / "results"
            for arm, mixed, base_mixed in (("RAVQC_W2", w2_mixed, 0.00203), ("RAVQC_W3", 0.00062, 0.00060)):
                write(g / "ravqc" / arm / "arm.json", dict(tuned=True))
                o = dict(mixed_loss=mixed, future_latent_nmse=0.03, first10motor_rmse=0.02, first10gripper_rmse=0.04)
                write(g / "ravqc" / "heldout" / f"{arm}.json", dict(overall=o))
                write(g / "heldout" / f"{p['arms'][arm]['base']}.json", dict(overall=dict(o, mixed_loss=base_mixed)))
            write(g / "ravqc" / "DECISION.json", dict(loops=[]))
            write(tmp / "protocol.json", p)
            out = subprocess.run([sys.executable, str(HERE / "gate.py"), "--protocol", str(tmp / "protocol.json"),
                                  "--wait", "5"], capture_output=True, text=True, check=True).stdout
            got = subprocess.run(["bash", "-c", parse], env=dict(GATE_OUT=out, PATH="/usr/bin:/bin"),
                                 capture_output=True, text=True, check=True).stdout.split()
            assert got == [str(len(want))] + want, (out, got)
            assert json.loads((tmp / "protocol.json").read_text(encoding="utf-8"))["arm_order"] == want
            if want == ["RAVQU_W2", "RAVQU_W3"]:  # calibrated arms only, none passing: an empty RUN line
                write(tmp / "protocol.json", dict(p, arm_order=["RAVQC_W2", "RAVQC_W3"]))
                out = subprocess.run([sys.executable, str(HERE / "gate.py"), "--protocol", str(tmp / "protocol.json"),
                                      "--wait", "5"], capture_output=True, text=True, check=True).stdout
                got = subprocess.run(["bash", "-c", parse], env=dict(GATE_OUT=out, PATH="/usr/bin:/bin"),
                                     capture_output=True, text=True, check=True).stdout.split()
                assert got == ["0"], (out, got)


def test_anchor():
    """The install check's reference: a calibrated arm's C80 fit row, an uncalibrated arm's panel-A loop startup; a
    startup that did not pass (or belongs to another arm) is refused."""
    sys.modules.setdefault("loop_worker", types.ModuleType("loop_worker"))
    common = types.ModuleType("common")
    common.TARGETS, common.guard, common.read, common.results, common.save = 0, None, gate.read, sched.results, gate.save
    sys.modules.setdefault("common", common)
    import panel_worker as pw
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        p = fake(tmp)
        write(tmp / "a16_protocol.json", dict(panel=dict(anchor_case=2000)))
        p["a16"]["protocol"] = str(tmp / "a16_protocol.json")
        g = tmp / "guided" / "results"
        write(g / "ravqc" / "RAVQC_W2" / "fit_after.json", dict(rows=[dict(case_id=2001, first10motor_rmse=0.5),
                                                                       dict(case_id=2000, first10motor_rmse=0.0219)]))
        start = dict(arm="RAVQU_W2", passed=True, anchor_action_vs_bf16=dict(first10_motor_rmse=0.0216))
        write(g / "loops" / "RAVQU_W2" / "worker_0" / "startup.json", start)
        assert pw.anchor_reference(p, "RAVQC_W2") == 0.0219
        assert pw.anchor_reference(p, "RAVQU_W2") == 0.0216
        for bad in (dict(start, passed=False), dict(start, arm="RAVQU_W3")):
            write(g / "loops" / "RAVQU_W2" / "worker_0" / "startup.json", bad)
            try:
                pw.anchor_reference(p, "RAVQU_W2")
            except RuntimeError:
                continue
            raise AssertionError(f"accepted {bad}")


def test_sched():
    with tempfile.TemporaryDirectory() as t:
        p = fake(Path(t))
        assert [sched.total(p, a) for a in p["arm_order"]] == [560, 560, 840, 840]
        sched.init(p)
        picks = [sched.choose(p, s) for s in range(4)]
        assert picks == ["RAVQC_W2", "RAVQC_W3", "RAVQU_W2", "RAVQU_W3"], picks
        ravqu = dict(p, arm_order=["RAVQU_W2", "RAVQU_W3"])  # the gate's copy when no calibrated arm passed
        sched.init(ravqu)
        assert sorted(sched.choose(ravqu, s) for s in range(4)) == ["RAVQU_W2", "RAVQU_W2", "RAVQU_W3", "RAVQU_W3"]
        one = dict(p, arm_order=["RAVQC_W2"])
        sched.init(one)
        assert [sched.choose(one, s) for s in range(4)] == ["RAVQC_W2"] * 4
        assert [len(panel.load(p, n)) for n in panel.PANELS] == [280, 280, 280]


def test_analysis():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        p = fake(tmp)
        rng = random.Random(0)
        job = Path(p["remote_root"]) / "results" / "loops"
        for arm, rate in (("RAVQC_W2", 0.74), ("RAVQC_W3", 0.75), ("RAVQU_W2", 0.73), ("RAVQU_W3", 0.74)):
            for i in range(280 if arm.startswith("RAVQU") else 0, 840):
                write(job / arm / "episodes" / f"{i}.json", dict(success=rng.random() < rate, policy_call_seconds=[0.5],
                                                                 invalid=i == 600))
        for arm, w in p["references"].items():
            for i in range(280):
                rec = dict(success=rng.random() < 0.72, index=i, arm="bf16", complete=True)
                if w["weights"] is None:
                    path = tmp / "rot" / "worker_0" / "episodes.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                else:
                    write(tmp / "a16_loops" / w["weights"] / "episodes" / f"{i}.json", rec)
            for i in range(280, 560):
                where = "panelc_loops" if arm.endswith("_W3") else "panelb_loops"
                write(tmp / where / arm / "episodes" / f"{i}.json", dict(success=rng.random() < 0.72))
            for i in range(560, 700 if arm == "QTIP_W2" else 840):  # QTIP_W2 panel C still running
                write(tmp / "panelc_loops" / arm / "episodes" / f"{i}.json", dict(success=rng.random() < 0.72,
                                                                                    invalid=i == 600))
        for arm in ("RAVQU_W2", "RAVQU_W3"):
            for i in range(280):
                write(tmp / "guided" / "results" / "loops" / arm / "episodes" / f"{i}.json", dict(success=i % 4 > 0))
        s = an.analyse(p)
        assert s["invalid_slots"] == [600] and s["valid"] == 839
        assert s["arms"]["RAVQC_W2"]["total"]["episodes"] == 839
        u2 = s["arms"]["RAVQU_W2"]
        assert u2["total"]["episodes"] == 839 and u2["panels"]["A"] == dict(successes=210, episodes=280), u2
        t2 = s["pairs"]["RAVQC_W2|QTIP_W2"]["total"]
        assert t2["slots"] == 280 + 280 + 139, t2
        assert s["pairs"]["RAVQU_W2|QTIP_W2"]["total"]["slots"] == 280 + 280 + 139
        assert not s["complete"]["QTIP_W2"]["C"] and s["complete"]["GPTVQ_W2"]["C"]
        assert s["pairs"]["RAVQC_W2|RAVQU_W2"]["total"]["slots"] == 839
        assert s["pairs"]["RAVQC_W3|RAVQU_W3"]["total"]["slots"] == 839
        assert "RAVQC_W3|QTIP_W3" in s["pairs"] and "RAVQC_W3|QTIP_W2" not in s["pairs"]
        assert not any(k.startswith("RAVQU_W2|RAVQ") for k in s["pairs"]), sorted(s["pairs"])
        text = (Path(p["remote_root"]) / "results" / "RESULTS.zh.md").read_text(encoding="utf-8")
        assert "RAVQC_W2 对 QTIP_W2" in text and "RAVQU_W2 对 QTIP_W2" in text and "（未全）" in text
        only = an.analyse(dict(p, arm_order=["RAVQC_W2"]))  # a RAVQU base the job did not run: panel A only
        assert only["pairs"]["RAVQC_W2|RAVQU_W2"]["total"]["slots"] == 280


if __name__ == "__main__":
    for test in (test_gate, test_gate_cli, test_anchor, test_sched, test_analysis):
        test()
        print("PASS", test.__name__, flush=True)
    print("ALL PASS")
