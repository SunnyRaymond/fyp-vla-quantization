"""Controller over the bounded, host-key-verified PBS base (upload / submit / status / logs / pull): --phase hosts is
the g2 two-GPU job of host.json (run.pbs), --phase rows the g1 one-GPU job of rows.json (run_rows.pbs), --phase gate
the g2 job of gate.json (run_gate.pbs), --phase full its g3 full run (run_full.pbs),
--phase tune the g2 three-GPU job of tune.json (run_tune.pbs), --phase tune1 / tune2 / tune3 / tune4 its split (g1 one GPU
on QTIP, g2 two GPUs on GPTVQ + QuIP#, g1 on GPTVQ, g1 on QuIP#; same script), --phase tuneloop its g3 closed loop (run_tune_loop.pbs); attempt "" or rN
(resubmission, resumes from the files on disk)."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "fwgh_control_base", HERE.parent / "fastwam-vq-r128-c80-fused-tuning/control.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
H = json.loads((HERE / "host.json").read_text(encoding="utf-8"))
ARMS = list(H["hosts"]["arms"])  # phase 3; phases 1 and 2 are in pulled/26043905.pbs101 and pulled/26046240.pbs101
SMALL = (["hosts/summary.json", "hosts/RESULTS.zh.md", "hosts/check/summary.json",
          "hosts/check/loop/results/episode.json", "hosts/check/loop/results/startup/startup.json"]
         + [p for arm in ARMS for p in (f"quantized/{arm}/COMPLETE.json", f"loops/{arm}/summary.json",
                                         f"loops/{arm}/RESULTS.zh.md")]
         + [f"heldout/{arm}.json" for arm in H["hosts"]["eval_arms"]]
         + [f"progress/fwgh{a}_{k}.json" for a in ("", "r3", "r4", "r5") for k in range(H["allocation"]["gpus"])])
ROWS = json.loads((HERE / "rows.json").read_text(encoding="utf-8"))
ROWS_SMALL = (["rows/summary.json", "rows/RESULTS.zh.md"]
              + [p.format(arm=ROWS["arm"]) for p in ("quantized/{arm}/COMPLETE.json", "heldout/{arm}.json",
                                                     "loops/{arm}/summary.json", "loops/{arm}/RESULTS.zh.md")]
              + [f"progress/fwgr{a}_0.json" for a in ("", "r1", "r2", "r3")])
GATE = json.loads((HERE / "gate.json").read_text(encoding="utf-8"))
GATE_SMALL = (["gate/summary.json", "gate/RESULTS.zh.md", "gate/DECISION.json", "gate/selection.json"]
              + [f"gate/{h}/{n}" for h in GATE["hosts"] for n in ("sigma.json", "ucheck.json")]
              + [f"gate/{h}/stats/{c}.json" for h in GATE["hosts"] for c in range(2000, 2080)]
              + [f"heldout/GATE_{h}_{v}.json" for h in GATE["hosts"] for v in ("G", "O")]
              + [f"progress/fwgo{a}_{k}.json" for a in ("", "r1", "r2", "r3") for k in range(GATE["allocation"]["gpus"])])
FULL_ARMS = [h + GATE["full"]["arm_suffix"] for h in GATE["hosts"]]
FULL_SMALL = (["full/summary.json", "full/RESULTS.zh.md", "gate/DECISION.json"]
              + [p for arm in FULL_ARMS for p in (f"quantized/{arm}/COMPLETE.json", f"heldout/{arm}.json",
                                                   f"loops/{arm}/summary.json", f"loops/{arm}/RESULTS.zh.md")]
              + [f"progress/fwgf{a}_{k}.json" for a in ("", "r1", "r2", "r3")
                 for k in range(GATE["full"]["allocation"]["gpus"])])
TUNE = json.loads((HERE / "tune.json").read_text(encoding="utf-8"))
TUNE_ARMS = [h + TUNE["arm_suffix"] for h in TUNE["hosts"]]
TUNE_SMALL = (["tune/summary.json", "tune/RESULTS.zh.md", "tune/DECISION.json"]
              + [f"tune/{h}/{n}" for h in TUNE["hosts"] for n in ("host.json", "lr.json", "step.json", "training.json",
                                                                 "fit_before.json", "fit_after.json")]
              + [p for arm in TUNE_ARMS for p in (f"quantized/{arm}/COMPLETE.json", f"heldout/{arm}.json")]
              + [f"progress/fwgt{s}{a}_{k}.json" for s in ("", "1", "2", "3", "4") for a in ("", "r1", "r2", "r3", "r4")
                 for k in range(TUNE["allocation"]["gpus"])])
TUNE_LOOP_SMALL = (["tune/loop/summary.json", "tune/loop/RESULTS.zh.md", "tune/DECISION.json"]
                   + [p for arm in TUNE_ARMS for p in (f"loops/{arm}/summary.json", f"loops/{arm}/RESULTS.zh.md")]
                   + [f"progress/fwgtl{a}_0.json" for a in ("", "r1", "r2", "r3")])
LOWRANK = json.loads((HERE / "lowrank.json").read_text(encoding="utf-8"))
LOWRANK_SMALL = (["lowrank/summary.json", "lowrank/RESULTS.zh.md", "lowrank/DECISION.json", "lowrank/init/layers.json"]
                 + [f"lowrank/{v['arm']}/{n}" for v in LOWRANK["arms"].values()
                    for n in ("arm.json", "step.json", "training.json", "fit.json", "COMPLETE.json")]
                 + [f"heldout/{v['arm']}.json" for v in LOWRANK["arms"].values()]
                 + [f"lowrank/QTIP_LRJ_W2/{n}.r0_26060556.json" for n in ("step", "arm")]
                 + [f"progress/{s}{a}_{k}.json" for s in ("fwlr1", "fwlr2", "fwlj1") for a in ("", "r1", "r2", "r3")
                    for k in range(2)])
LOWRANK_DIAG_SMALL = (["lowrank/diag/summary.json", "lowrank/diag/RESULTS.zh.md", "lowrank/diag/teacher_check.json"]
                      + [f"lowrank/diag/fit_{LOWRANK['arms'][k]['arm']}_s{LOWRANK['diag']['seed']}.json"
                         for k in LOWRANK["diag"]["fit_arms"]]
                      + [f"progress/fwld1{a}_0.json" for a in ("", "r1", "r2", "r3")])
LOWRANK_TF_SMALL = (["lowrank/tfdiag/summary.json", "lowrank/tfdiag/RESULTS.zh.md", "lowrank/tfdiag/state_check.json"]
                    + [f"progress/fwlt1{a}_0.json" for a in ("", "r1", "r2", "r3")])
LOWRANK_LOOP_ARM = LOWRANK["arms"][LOWRANK["loop"]["arm"]]["arm"]
LOWRANK_LOOP_NAME = LOWRANK_LOOP_ARM + ("" if LOWRANK["loop"]["mode"] == "branch" else "_" + LOWRANK["loop"]["mode"])
LOWRANK_LOOP_SMALL = ([f"loops/{LOWRANK_LOOP_NAME}/{n}" for n in ("summary.json", "RESULTS.zh.md", "paired.json", "DONE")]
                      + [f"loops/{LOWRANK_LOOP_NAME}/worker_{w}/{n}"
                         for w in range(LOWRANK["loop"]["allocation"]["gpus"] * LOWRANK["loop"]["workers_per_gpu"])
                         for n in ("startup.json", "lowrank_check.json", "summary.json")])
XFER = LOWRANK["xfer"]
XFER_PROGRESS = [f"progress/{{}}{a}_{k}.json" for a in ("", "r1", "r2", "r3") for k in range(len(XFER["hosts"]))]
LOWRANK_X1_SMALL = ([f"lowrank/{h}_LRE_W2/{n}" for h in XFER["hosts"] for n in ("arm.json", "COMPLETE.json", "fit.json")]
                    + [f"heldout/{h}_LRE_W2.json" for h in XFER["hosts"]]
                    + [f"lowrank/init_{h}/layers.json" for h in XFER["hosts"]]
                    + [p.format("fwlx1") for p in XFER_PROGRESS])
LOWRANK_X2_SMALL = ([f"lowrank/{h}_LRJ_W2/{n}" for h in XFER["hosts"]
                     for n in ("arm.json", "step.json", "training.json", "COMPLETE.json", "fit.json")]
                    + [f"heldout/{h}_LRJ_W2.json" for h in XFER["hosts"]]
                    + ["lowrank/xfer/summary.json", "lowrank/xfer/RESULTS.zh.md"]
                    + [p.format("fwlx2") for p in XFER_PROGRESS])
LOWRANK_XL_SMALL = (["lowrank/xfer/loops.json"]
                    + [f"loops/{h}_LRJ_W2/{n}" for h in XFER["hosts"] for n in ("summary.json", "RESULTS.zh.md", "paired.json", "DONE")]
                    + [f"loops/{h}_LRJ_W2/worker_{w}/{n}" for h in XFER["hosts"]
                       for w in range(XFER["loop"]["gpus_per_host"] * XFER["loop"]["workers_per_gpu"])
                       for n in ("startup.json", "lowrank_check.json", "summary.json")])
LOWRANK_BENCH_SMALL = (["lowrank/bench/summary.json"]
                       + [f"lowrank/bench/{m}/{n}" for m in ("branch", "fused", "merged", "host")
                          for n in ("bench.json", "lowrank_check.json", "startup.json")])
LOWRANK_TC_SMALL = (["lowrank/tcond/summary.json", "lowrank/tcond/RESULTS.zh.md", "lowrank/tcond/input_check.json"]
                    + [f"heldout/{v['arm']}.json" for v in LOWRANK["tcond"]["arms"].values()]
                    + [f"progress/fwtc1{a}_0.json" for a in ("", "r1", "r2", "r3")])
CONFIG = {"hosts": (H["allocation"], "run.pbs", SMALL), "rows": (ROWS["allocation"], "run_rows.pbs", ROWS_SMALL),
          "gate": (GATE["allocation"], "run_gate.pbs", GATE_SMALL),
          "full": (GATE["full"]["allocation"], "run_full.pbs", FULL_SMALL),
          "tune": (TUNE["allocation"], "run_tune.pbs", TUNE_SMALL),
          "tune1": (TUNE["split"]["one"], "run_tune.pbs", TUNE_SMALL),
          "tune2": (TUNE["split"]["two"], "run_tune.pbs", TUNE_SMALL),
          "tune3": (TUNE["split"]["gptvq"], "run_tune.pbs", TUNE_SMALL),
          "tune4": (TUNE["split"]["quipsharp"], "run_tune.pbs", TUNE_SMALL),
          "tuneloop": (TUNE["loop"]["allocation"], "run_tune_loop.pbs", TUNE_LOOP_SMALL),
          "lowrank1": (LOWRANK["split"]["one"], "run_lowrank.pbs", LOWRANK_SMALL),
          "lowrank2": (LOWRANK["split"]["two"], "run_lowrank.pbs", LOWRANK_SMALL),
          "lowrankj": (LOWRANK["split"]["j"], "run_lowrank.pbs", LOWRANK_SMALL),
          "lowrankd": (LOWRANK["diag"]["allocation"], "run_lowrank.pbs", LOWRANK_DIAG_SMALL),
          "lowrankt": (LOWRANK["tfdiag"]["allocation"], "run_lowrank.pbs", LOWRANK_TF_SMALL),
          "lowrankc": (LOWRANK["tcond"]["allocation"], "run_lowrank.pbs", LOWRANK_TC_SMALL),
          "lowrankl": (LOWRANK["loop"]["allocation"], "run_lowrank_loop.pbs", LOWRANK_LOOP_SMALL),
          "lowrankb": (LOWRANK["loop"]["bench"]["allocation"], "run_lowrank_loop.pbs", LOWRANK_BENCH_SMALL),
          "lowrankx1": (XFER["init"]["allocation"], "run_lowrank_xfer.pbs", LOWRANK_X1_SMALL),
          "lowrankx2": (XFER["train"]["allocation"], "run_lowrank_xfer.pbs", LOWRANK_X2_SMALL),
          "lowrankxl": (XFER["loop"]["allocation"], "run_lowrank_xfer.pbs", LOWRANK_XL_SMALL)}


def configure(attempt="", phase="hosts"):
    if not re.fullmatch(r"(r\d)?", attempt):
        raise ValueError(f"Unknown attempt: {attempt!r}")
    a, script, _ = CONFIG[phase]
    base.HERE, base.PROTOCOL_ID, base.ROOT = HERE, P["protocol"], P["remote_root"]
    base.JOB_NAME = a["job_name"] + attempt
    base.JOB_NAME_RE = re.compile("^" + re.escape(base.JOB_NAME) + "$")
    base.ALLOCATION = {k: a[k] for k in ("queue", "execution_queue", "gpus", "ncpus", "memory_gb", "walltime")}
    base.QUEUE, base.EXECUTION_QUEUE, base.WALLTIME = a["queue"], a["execution_queue"], a["walltime"]
    base.SELECT = f"1:ncpus={a['ncpus']}:mem={a['memory_gb']}gb:ngpus={a['gpus']}"
    base.ARTIFACTS, base.RESULTS = base.ROOT + "/artifacts", base.ROOT + "/results"
    base.STATE_PATH = base.ROOT + f"/control/state{'' if phase == 'hosts' else '_' + phase}{attempt}.json"
    base.FILES = ("protocol.json", "run.pbs", "environment.sh", "worker.py", "probe.py", "host.json", "loop_check.py",
                  "rowcheck.py", "rowcheck.pbs", "test_probe.py", "test_hosts.py", "cputest.pbs",
                  "rows.json", "rows_worker.py", "run_rows.pbs", "test_rows.py",
                  "gate.json", "gate_worker.py", "run_gate.pbs", "test_gate.py", "run_full.pbs",
                  "tune.json", "tune_worker.py", "run_tune.pbs", "run_tune_loop.pbs", "test_tune.py",
                  "lowrank.json", "lowrank_worker.py", "run_lowrank.pbs", "test_lowrank.py", "tfpool.py", "tfpool.pbs",
                  "lowrank_loop.py", "lowrank_bench.py", "run_lowrank_loop.pbs", "test_lowrank_loop.py",
                  "run_lowrank_xfer.pbs")
    base._qsub_command = lambda: (
        "qsub -N " + base.JOB_NAME + " -q " + base.QUEUE + " -P " + base.PROJECT + " -l select=" + shlex.quote(base.SELECT)
        + " -l walltime=" + base.WALLTIME + " -o " + shlex.quote(base.ARTIFACTS + "/" + base.JOB_NAME + ".pbs.out")
        + " -j oe " + shlex.quote(base.ROOT + "/" + script))

    def save_state(transport, state, _original=base._save_state):
        if state.get("handle"):
            state["handle"].update(gpus=a["gpus"], ncpus=a["ncpus"], memory_gb=a["memory_gb"])
        _original(transport, state)
    base._save_state = save_state
    return a


def load_protocol():
    value = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    if value.get("protocol") != base.PROTOCOL_ID or value.get("remote_root") != base.ROOT:
        raise RuntimeError("protocol.json identity differs")
    return value


def complete_gate(value):
    h, s, a = (value.get(k) or {} for k in ("handle", "scheduler", "artifact"))
    return (h.get("protocol") == P["protocol"] and h.get("job_name") == base.JOB_NAME
            and s.get("query_rc") == 0 and s.get("job_state") == "F" and s.get("Exit_status") == "0"
            and s.get("Job_Name") == base.JOB_NAME and s.get("queue") == base.EXECUTION_QUEUE
            and a.get("exit_code") == "0" and a.get("pipeline_complete") is True)


base.load_protocol, base.complete_gate = load_protocol, complete_gate


def status(transport, gpus):
    value = base.status(transport)
    if value.get("handle"):
        with base.remote.sftp_client(transport) as sftp:
            value["workers"] = {
                str(k): dict(progress=base._json(sftp, base.RESULTS + f"/progress/{base.JOB_NAME}_{k}.json"),
                             failure=base._json(sftp, base.RESULTS + f"/failures/{base.JOB_NAME}_{k}.json"))
                for k in range(gpus)}
    return value


def pull(transport, gpus, small=SMALL):
    value = status(transport, gpus)
    if not value.get("handle"):
        raise RuntimeError("No submitted job")
    target = HERE / "pulled" / value["handle"]["job_id"]
    files = []
    with base.remote.sftp_client(transport) as sftp:
        for name in small:
            data = base._small_bytes(sftp, base.RESULTS + "/" + name, limit=16 * 1024 * 1024)
            if data is not None:
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                files.append(name)
    (HERE / "monitor_status.json").write_text(
        json.dumps({k: v for k, v in value.items() if k != "log_tail"}, indent=2, default=str) + "\n",
        encoding="utf-8")
    return dict(directory=str(target), files=len(files), status=value.get("status"))


def cpu_submit(transport, script, name, after=None):
    """Plain qsub of a CPU-node script (2 cores, 16 GB; walltime > 2 h routes to q2 instead of the slow qdev); after:
    "<id>[:<id> ...]" adds -W depend=afterok (starts only once every listed job exited 0)."""
    if after is not None and not re.fullmatch(r"\d+\.pbs101(:\d+\.pbs101)*", after):
        raise ValueError(f"Bad PBS job id list for afterok: {after!r}")
    command = ("qsub -N " + name + (f" -W depend=afterok:{after}" if after else "") + " -q normal -P " + base.PROJECT
               + " -l select=1:ncpus=2:mem=16gb -l walltime=02:30:00"
               + " -o " + base.ARTIFACTS + f"/{name}.pbs.out -j oe " + P["remote_root"] + "/" + script)
    code, out = base.remote.command(transport, command, timeout=45)
    if code or not base.HANDLE.fullmatch(out.strip()):
        raise RuntimeError("qsub failed: " + out[-500:])
    return dict(job_id=out.strip())


def rowcheck_status(transport, job_id):
    scheduler = base._scheduler(transport, job_id)
    with base.remote.sftp_client(transport) as sftp:
        data = base._small_bytes(sftp, base.RESULTS + "/rowcheck/summary.json", limit=16 * 1024 * 1024)
    if data is not None:
        target = HERE / "pulled" / "rowcheck" / "summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        data = json.loads(data)["summary"]
    return dict(job_state=scheduler.get("job_state"), exit_status=scheduler.get("Exit_status"), summary=data)


def tfpool_status(transport, job_id):
    scheduler = base._scheduler(transport, job_id)
    with base.remote.sftp_client(transport) as sftp:
        data = base._small_bytes(sftp, base.RESULTS + "/lowrank/tfdiag/pool.json", limit=16 * 1024 * 1024)
    if data is not None:
        target = HERE / "pulled" / "tfpool" / "pool.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return dict(job_state=scheduler.get("job_state"), exit_status=scheduler.get("Exit_status"), pulled=data is not None)


def cputest_status(transport, job_id):
    scheduler = base._scheduler(transport, job_id)
    found = {}
    with base.remote.sftp_client(transport) as sftp:
        for name in ("test_probe", "test_hosts", "test_rows", "test_gate", "test_tune", "test_lowrank", "test_lowrank_loop"):
            for ext in ("json", "log"):
                data = base._small_bytes(sftp, f"{base.RESULTS}/cputest/{job_id}/{name}.{ext}")
                if data is not None:
                    text = data.decode("utf-8", errors="replace")
                    found[f"{name}.{ext}"] = json.loads(text) if ext == "json" else text[-3000:]
        found["passed"] = base._small_bytes(sftp, f"{base.RESULTS}/cputest/{job_id}/PASSED") is not None
    return dict(job_state=scheduler.get("job_state"), exit_status=scheduler.get("Exit_status"), **found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="")
    parser.add_argument("--phase", choices=tuple(CONFIG), default="hosts")
    parser.add_argument("--id")
    parser.add_argument("action", choices=("upload", "submit", "status", "logs", "pull", "rowcheck", "rowcheck-status",
                                           "cputest", "cputest-status", "tfpool", "tfpool-status"))
    args = parser.parse_args()
    gpus = configure(args.attempt, args.phase)["gpus"]
    jump, transport = base.direct_connect()
    try:
        if args.action == "rowcheck":
            result = {"uploaded": base.upload(transport), **cpu_submit(transport, "rowcheck.pbs", "fwghrc")}
        elif args.action == "rowcheck-status":
            result = rowcheck_status(transport, args.id)
        elif args.action == "cputest":
            result = {"uploaded": base.upload(transport), **cpu_submit(transport, "cputest.pbs", "fwghct")}
        elif args.action == "cputest-status":
            result = cputest_status(transport, args.id)
        elif args.action == "tfpool":
            result = {"uploaded": base.upload(transport), **cpu_submit(transport, "tfpool.pbs", "fwgtp")}
        elif args.action == "tfpool-status":
            result = tfpool_status(transport, args.id)
        elif args.action == "upload":
            result = {"uploaded": base.upload(transport)}
        elif args.action == "submit":
            result = base.submit(transport)
        elif args.action == "logs":
            result = base.logs(transport)
        elif args.action == "pull":
            result = pull(transport, gpus, CONFIG[args.phase][2])
        else:
            result = status(transport, gpus)
            result.pop("log_tail", None) if result.get("status") == "queued" else None
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    main()
