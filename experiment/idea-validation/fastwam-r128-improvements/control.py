"""Small-file, host-key-verified PBS control for the frozen R128 lanes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "fastwam-libero-plus-pilot"
REMOTE_HELPER = PILOT / "remote.py"
ROOT = "/scratch/users/ntu/yguo017/fastwam-r128-improvements-20261021"
PROTOCOL_ID = "fastwam-r128-improvements-v1"
USER = "yguo017"
PROJECT = "personal-yguo017"
SUBMISSION_QUEUE = "normal"
LIMIT = 256 * 1024
MAX_BYTES = LIMIT - 1
HANDLE = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")
KINDS = ("path", "optimization", "coverage")
JOB_NAMES = {"path": "fwr128path", "optimization": "fwr128opt",
             "coverage": "fwr128coverage"}
CODE_FILES = ("protocol.json", "run.pbs", "environment.sh", "queue_routing.json")
RESULT_FILES = ("summary.json", "progress.json", "selfcheck.json", "training.json",
                "comparison.json", "RESULTS.zh.md")
RESULT_JSON_FILES = tuple(name for name in RESULT_FILES if name.endswith(".json"))
RUNTIME_FILES = (
    "runner.py", "quantization.py", "libero_state_loader.py", "smooth_vq.py",
    "codebook.py", "micro_codebook.py", "sampler.py", "latent_reference.py",
    "checkpointing.py", "calibration_source.py", "c8_train_banks.py", "batched_transform.py",
)


def load_protocol() -> dict:
    path = HERE / "protocol.json"
    if not path.is_file() or path.stat().st_size >= LIMIT:
        raise RuntimeError("protocol.json is missing or exceeds the 256 KiB control-file limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol") != PROTOCOL_ID:
        raise RuntimeError("protocol.json does not match the frozen R128 improvements protocol")
    if value.get("remote_root") != ROOT:
        raise RuntimeError("protocol.json names an unexpected remote root")
    jobs = value.get("jobs")
    lanes = value.get("lanes")
    if not isinstance(jobs, dict) or not isinstance(lanes, dict):
        raise RuntimeError("protocol.json lacks frozen jobs/lanes maps")
    expected = {"path": (["path"], 1, "g1"),
                "optimization": (["optimization"], 1, "g1"),
                "coverage": (["coverage_c80"], 1, "g1")}
    if set(jobs) != set(expected):
        raise RuntimeError("protocol job kinds differ from the frozen single-GPU and paired coverage jobs")
    seen_lanes = set()
    for kind, (expected_lanes, expected_gpus, expected_queue) in expected.items():
        job = jobs.get(kind)
        if (not isinstance(job, dict) or job.get("gpus") != expected_gpus
                or job.get("lanes") != expected_lanes or job.get("execution_queue") != expected_queue):
            raise RuntimeError(f"protocol has an unexpected lane/GPU/queue mapping for {kind}")
        if any(not isinstance(lanes.get(lane), dict) for lane in expected_lanes):
            raise RuntimeError(f"protocol is missing a lane plan for {kind}")
        seen_lanes.update(expected_lanes)
    if seen_lanes != set(lanes):
        raise RuntimeError("protocol lane plans contain missing or unexpected lanes")
    if value.get("allocation", {}).get("queue") != SUBMISSION_QUEUE:
        raise RuntimeError("protocol must submit through the verified normal routing queue")
    return value


def load_remote_helper():
    sys.path.insert(0, str(PILOT))
    spec = importlib.util.spec_from_file_location("fastwam_r128_improvements_remote", REMOTE_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trusted ASPIRE2A helper: {REMOTE_HELPER}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_local(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def state_path(kind: str) -> Path:
    return HERE / f"control_state_{kind}.json"


def remote_state_path(kind: str) -> str:
    return f"{ROOT}/control/state_{kind}.json"


def read_local_state(kind: str) -> dict:
    path = state_path(kind)
    if not path.is_file():
        return {}
    if path.stat().st_size >= LIMIT:
        raise RuntimeError(f"Local {kind} control state exceeds the 256 KiB limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Local {kind} control state must be a JSON object")
    return value


def write_remote_json(sftp, path: str, value: dict) -> None:
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    if len(data) >= LIMIT:
        raise RuntimeError("Remote control JSON exceeds the 256 KiB limit")
    with sftp.file(path + ".tmp", "wb") as stream:
        stream.write(data)
    sftp.posix_rename(path + ".tmp", path)


def read_small_bytes(sftp, path: str, limit: int = MAX_BYTES) -> bytes | None:
    try:
        attrs = sftp.stat(path)
    except OSError as exc:
        if getattr(exc, "errno", None) == 2:
            return None
        raise
    if attrs.st_size > limit:
        raise RuntimeError(f"Refusing control/result file larger than {limit} bytes: {path}")
    with sftp.file(path, "rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError(f"Remote file grew beyond its bounded read: {path}")
    return data


def read_remote_json(sftp, path: str) -> dict | None:
    data = read_small_bytes(sftp, path)
    if data is None:
        return None
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def local_files() -> list[Path]:
    required = ("path_worker.py", "train_worker.py", "rotation_bank.py",
                "rotation_selfcheck.py", "aggregate.py", *CODE_FILES)
    missing = [name for name in required if not (HERE / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing staged sources: " + ", ".join(missing))
    paths = sorted(HERE.glob("*.py"), key=lambda path: path.name)
    paths.extend(HERE / name for name in CODE_FILES)
    for path in paths:
        if path.parent.resolve() != HERE.resolve() or not path.is_file():
            raise FileNotFoundError(f"Missing local control file: {path.name}")
        if path.stat().st_size >= LIMIT:
            raise RuntimeError(f"Only small code/control files below 256 KiB may be staged: {path.name}")
        path.read_bytes().decode("utf-8")
    return paths


def normalized_local_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def transfer_files(sftp) -> list[dict]:
    sent = []
    for path in local_files():
        data = normalized_local_bytes(path)
        target = ROOT + "/" + path.name
        with sftp.file(target + ".tmp", "wb") as stream:
            stream.write(data)
        sftp.posix_rename(target + ".tmp", target)
        sent.append({"name": path.name, "bytes": len(data)})
    return sent


def assert_remote_files(sftp) -> None:
    paths = local_files()
    local_python = {path.name for path in paths if path.suffix == ".py"}
    remote_python = {name for name in sftp.listdir(ROOT) if name.endswith(".py")}
    if remote_python != local_python:
        raise RuntimeError("Remote Python snapshot contains missing or unexpected files")
    for path in paths:
        if read_small_bytes(sftp, ROOT + "/" + path.name) != normalized_local_bytes(path):
            raise RuntimeError(f"Remote file is missing or differs from the frozen local file: {path.name}")


def merged_state(remote_state: dict, local_state: dict) -> dict:
    state = {**local_state, **remote_state}
    for key in ("intent", "handle", "last_terminal", "last_verified"):
        if not state.get(key):
            state[key] = local_state.get(key) or remote_state.get(key)
    return state


def save_state(remote, transport, kind: str, state: dict) -> None:
    with remote.sftp_client(transport) as sftp:
        write_remote_json(sftp, remote_state_path(kind), state)
    save_local(state_path(kind), state)


def job_name(kind: str, attempt: int) -> str:
    if kind not in KINDS or type(attempt) is not int or attempt < 0:
        raise ValueError("Invalid job kind or attempt number")
    return JOB_NAMES[kind] if attempt == 0 else f"{JOB_NAMES[kind]}-a{attempt}"


def attempt_number(handle: dict, kind: str) -> int:
    attempt = handle.get("attempt")
    if type(attempt) is int and attempt >= 0:
        return attempt
    name = str(handle.get("job_name", ""))
    if name == job_name(kind, 0):
        return 0
    match = re.fullmatch(re.escape(JOB_NAMES[kind]) + r"-a([1-9][0-9]*)", name)
    if match:
        return int(match.group(1))
    raise RuntimeError("Saved PBS handle has no valid attempt number or job name")


def assert_handle_scope(handle: dict, kind: str) -> None:
    if handle.get("remote_root", ROOT) != ROOT:
        raise RuntimeError("Saved PBS handle belongs to a different remote root")
    if handle.get("protocol") not in (None, PROTOCOL_ID) or handle.get("job_kind") not in (None, kind):
        raise RuntimeError("Saved PBS handle belongs to a different protocol or lane")


def named_jobs(remote, transport, name: str) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Invalid PBS job name")
    code, output = remote.command(
        transport, f"qselect -x -u {USER} -N {shlex.quote(name)} 2>&1", timeout=35
    )
    if code not in (0, 1) or (code == 1 and output.strip()):
        raise RuntimeError(f"PBS history query failed for {name}; refusing submission")
    matches = output.split()
    if any(not HANDLE.fullmatch(job_id) for job_id in matches):
        raise RuntimeError(f"PBS returned an invalid job handle for {name}")
    return matches


def recover_intent(remote, transport, state: dict, kind: str) -> dict | None:
    intent = state.get("intent") or {}
    if intent.get("kind") not in ("submit", "resume") or intent.get("status") not in (
        "intent_before_qsub", "qsub_outcome_unknown",
    ):
        return None
    name = intent.get("job_name")
    if not isinstance(name, str) or name != job_name(kind, intent.get("attempt")):
        raise RuntimeError("Unresolved qsub intent has no valid candidate job name")
    matches = named_jobs(remote, transport, name)
    if len(matches) != 1:
        raise RuntimeError("qsub outcome is unresolved; PBS history did not identify exactly one job. Do not resubmit.")
    handle = {
        "job_id": matches[0], "job_name": name, "attempt": intent.get("attempt"),
        "remote_root": ROOT, "protocol": PROTOCOL_ID, "job_kind": kind,
        "status": "recovered_from_pbs_history", "confirmed_utc": utc_now(),
    }
    intent.update(status="confirmed", job_id=matches[0], recovered_utc=utc_now())
    state.update(intent=intent, handle=handle)
    return handle


def remote_protocol_matches(sftp, protocol: dict) -> None:
    if read_remote_json(sftp, ROOT + "/protocol.json") != protocol:
        raise RuntimeError("Remote protocol differs from the frozen local protocol")


def job_resources(kind: str, protocol: dict | None = None) -> tuple[str, str, str]:
    protocol = protocol or load_protocol()
    job = protocol["jobs"][kind]
    allocation = protocol["allocation"]
    gpus = job["gpus"]
    cpus = allocation["ncpus_per_gpu"] * gpus
    memory = allocation["memory_gb_per_gpu"] * gpus
    select = f"1:ncpus={cpus}:mem={memory}gb:ngpus={gpus}"
    return job["execution_queue"], select, allocation["walltime"]


def qsub_command(kind: str, name: str, attempt: int, previous_job_id: str | None = None) -> str:
    variables = f"FW_JOB_KIND={kind},FW_ATTEMPT={attempt}"
    if previous_job_id:
        variables += f",FW_PREVIOUS_JOBID={previous_job_id}"
    target_queue, select, walltime = job_resources(kind)
    return (
        f"qsub -N {shlex.quote(name)} -q {SUBMISSION_QUEUE} -P {PROJECT} "
        f"-l select={select} -l walltime={walltime} -v {shlex.quote(variables)} "
        f"-o {shlex.quote(ROOT + '/artifacts/' + name + '.pbs.out')} -j oe "
        f"{shlex.quote(ROOT + '/run.pbs')}"
    )


def stage(remote, transport, kind: str) -> dict:
    protocol = load_protocol()
    local_files()
    code, output = remote.command(
        transport, "mkdir -p " + " ".join(map(shlex.quote, (
            ROOT, ROOT + "/control", ROOT + "/artifacts", ROOT + "/results",
        ))), timeout=25,
    )
    if code:
        raise RuntimeError(f"Could not create remote control directories: {output[-400:]}")
    with remote.sftp_client(transport) as sftp:
        remote_protocol = read_remote_json(sftp, ROOT + "/protocol.json")
        if remote_protocol == protocol:
            try:
                assert_remote_files(sftp)
                return {"job_kind": kind, "staged": False, "already_staged": True,
                        "files": len(local_files()), "remote_root": ROOT}
            except (OSError, RuntimeError):
                pass
        for candidate in KINDS:
            remote_state = read_remote_json(sftp, remote_state_path(candidate)) or {}
            local_state = read_local_state(candidate)
            if any(value.get(key) for value in (remote_state, local_state) for key in ("handle", "intent")):
                raise RuntimeError("A job handle or qsub intent exists; refusing to change the staged snapshot")
        transfer_files(sftp)
        remote_protocol_matches(sftp, protocol)
        assert_remote_files(sftp)
    return {"job_kind": kind, "staged": True, "files": len(local_files()), "remote_root": ROOT}


def submit(remote, transport, kind: str) -> dict:
    protocol = load_protocol()
    remote_state = {}
    with remote.sftp_client(transport) as sftp:
        remote_state = read_remote_json(sftp, remote_state_path(kind)) or {}
    state = merged_state(remote_state, read_local_state(kind))
    recovered = recover_intent(remote, transport, state, kind)
    if recovered:
        save_state(remote, transport, kind, state)
        return {"recovered": True, "job_kind": kind, "handle": recovered}
    if state.get("handle"):
        handle = state["handle"]
        if not HANDLE.fullmatch(str(handle.get("job_id", ""))):
            raise RuntimeError("Saved PBS handle is invalid")
        assert_handle_scope(handle, kind)
        return {"existing": True, "job_kind": kind, "handle": handle}
    if state.get("intent"):
        raise RuntimeError("Unresolved submission intent exists; do not submit another job")
    name = job_name(kind, 0)
    with remote.sftp_client(transport) as sftp:
        remote_protocol_matches(sftp, protocol)
        assert_remote_files(sftp)
    matches = named_jobs(remote, transport, name)
    if len(matches) > 1:
        raise RuntimeError("Multiple jobs already use this lane job name; refusing submission")
    if matches:
        handle = {"job_id": matches[0], "job_name": name, "attempt": 0,
                  "remote_root": ROOT, "protocol": PROTOCOL_ID, "job_kind": kind,
                  "status": "recovered_from_pbs_history", "confirmed_utc": utc_now()}
        state.update(handle=handle, intent={"kind": "submit", "status": "confirmed",
                                           "job_name": name, "job_id": matches[0]})
        save_state(remote, transport, kind, state)
        return {"recovered": True, "job_kind": kind, "handle": handle}
    target_queue, select, walltime = job_resources(kind, protocol)
    intent = {"kind": "submit", "job_kind": kind, "attempt": 0, "job_name": name,
              "script": ROOT + "/run.pbs", "created_utc": utc_now(),
              "status": "intent_before_qsub", "queue": SUBMISSION_QUEUE,
              "target_queue": target_queue, "project": PROJECT,
              "select": select, "walltime": walltime}
    state.update(intent=intent, handle=None)
    save_state(remote, transport, kind, state)
    try:
        code, output = remote.command(transport, qsub_command(kind, name, 0), timeout=40)
    except Exception as exc:
        intent.update(status="qsub_outcome_unknown", error=type(exc).__name__, updated_utc=utc_now())
        save_state(remote, transport, kind, state)
        raise RuntimeError("qsub outcome is unknown; retry only to recover PBS history, never to resubmit") from exc
    job_id = output.strip()
    if code or not HANDLE.fullmatch(job_id):
        intent.update(status="qsub_outcome_unknown", response=output[:1000], return_code=code,
                      updated_utc=utc_now())
        save_state(remote, transport, kind, state)
        raise RuntimeError("qsub outcome is unknown or rejected; recover the exact PBS job before another submission")
    handle = {"job_id": job_id, "job_name": name, "attempt": 0, "remote_root": ROOT,
              "protocol": PROTOCOL_ID, "job_kind": kind, "status": "confirmed", "confirmed_utc": utc_now()}
    intent.update(status="confirmed", job_id=job_id, confirmed_utc=utc_now())
    state.update(intent=intent, handle=handle)
    save_state(remote, transport, kind, state)
    return {"job_kind": kind, "handle": handle}


def scheduler(remote, transport, job_id: str) -> dict:
    if not HANDLE.fullmatch(job_id):
        raise ValueError("Invalid PBS job handle")
    code, output = remote.command(transport, f"qstat -xf {shlex.quote(job_id)}", timeout=35)
    fields = {}
    for key in ("job_state", "Exit_status", "Job_Name", "queue", "exec_host", "comment",
                "resources_used.walltime"):
        match = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*(.*)$", output, re.M)
        fields[key] = match.group(1).strip() if match else None
    return {"job_id": job_id, "query_rc": code, **fields}


def retain_terminal(current: dict, saved: dict | None, job_id: str) -> tuple[dict, dict | None]:
    if current.get("query_rc") == 0:
        current = {**current, "terminal_from_history": False}
        if current.get("job_state") in ("F", "X") and current.get("Exit_status") is not None:
            saved = {key: current.get(key) for key in (
                "job_id", "query_rc", "job_state", "Exit_status", "Job_Name", "queue", "exec_host",
                "resources_used.walltime",
            )}
    elif isinstance(saved, dict) and saved.get("job_id") == job_id:
        current = {**saved, "current_query_rc": current.get("query_rc"), "terminal_from_history": True}
    return current, saved


def read_exit_code(sftp, path: str) -> int | None:
    data = read_small_bytes(sftp, path, 64)
    if data is None:
        return None
    match = re.fullmatch(rb"\s*(-?\d+)\s*", data)
    if not match:
        raise RuntimeError(f"Malformed exit code receipt: {path}")
    return int(match.group(1))


def artifact_observation(sftp, handle: dict, kind: str, protocol: dict) -> dict:
    job_id = handle["job_id"]
    artifact = ROOT + "/artifacts/" + job_id
    lane_observations = {}
    for lane in protocol["jobs"][kind]["lanes"]:
        lane_root = ROOT + "/results/" + lane
        lane_artifact = artifact if len(protocol["jobs"][kind]["lanes"]) == 1 else artifact + "/" + lane
        result_json = {name: read_remote_json(sftp, lane_root + "/" + name) for name in RESULT_JSON_FILES}
        result_files = {}
        for name in RESULT_FILES:
            try:
                sftp.stat(lane_root + "/" + name)
                result_files[name] = True
            except OSError as exc:
                if getattr(exc, "errno", None) != 2:
                    raise
                result_files[name] = False
        worker_summary = read_remote_json(sftp, lane_artifact + "/worker_summary.json")
        lane_exit_code = read_exit_code(sftp, lane_artifact + "/exit_code.txt")
        try:
            sftp.stat(lane_artifact + "/PHASE_COMPLETE")
            lane_marker = True
        except OSError as exc:
            if getattr(exc, "errno", None) != 2:
                raise
            lane_marker = False
        reuse_receipt = read_remote_json(sftp, lane_artifact + "/reuse_receipt.json")
        reuse_verified = False
        lane_summary = result_json.get("summary.json") or {}
        if isinstance(reuse_receipt, dict) and reuse_receipt.get("reused_completed_lane") is True:
            source_job = reuse_receipt.get("source_job_id")
            if (isinstance(source_job, str) and HANDLE.fullmatch(source_job)
                    and source_job == lane_summary.get("pbs_jobid")
                    and reuse_receipt.get("current_job_id") == job_id):
                source_artifact = ROOT + "/artifacts/" + str(source_job) + "/" + lane
                source_summary = read_remote_json(sftp, source_artifact + "/worker_summary.json")
                source_exit = read_exit_code(sftp, source_artifact + "/exit_code.txt")
                try:
                    sftp.stat(source_artifact + "/PHASE_COMPLETE")
                    source_marker = True
                except OSError as exc:
                    if getattr(exc, "errno", None) != 2:
                        raise
                    source_marker = False
                reuse_verified = (source_summary == lane_summary and source_exit == 0 and source_marker)
        checkpoint_sizes = {}
        checkpoint_paths = ([ ("state", lane_root + "/state.pt") ] if lane == "path" else [
            (arm["name"], lane_root + "/arms/" + arm["name"] + "/snapshot.pt")
            for arm in protocol["lanes"][lane].get("arms", []) if isinstance(arm, dict) and arm.get("name")
        ])
        for name, path in checkpoint_paths:
            try:
                checkpoint_sizes[name] = sftp.stat(path).st_size
            except OSError as exc:
                if getattr(exc, "errno", None) != 2:
                    raise
                checkpoint_sizes[name] = None
        lane_observations[lane] = {
            "lane": lane, "result_json": result_json, "result_files": result_files,
            "worker_summary": worker_summary, "artifact_exit_code": lane_exit_code,
            "phase_complete": lane_marker, "checkpoint_sizes": checkpoint_sizes,
            "checkpoint_size": sum(size for size in checkpoint_sizes.values() if isinstance(size, int)) or None,
            "reuse_receipt": reuse_receipt, "reuse_verified": reuse_verified,
        }
    worker_summary = read_remote_json(sftp, artifact + "/worker_summary.json")
    exit_code = read_exit_code(sftp, artifact + "/exit_code.txt")
    try:
        sftp.stat(artifact + "/PHASE_COMPLETE")
        marker = True
    except OSError as exc:
        if getattr(exc, "errno", None) != 2:
            raise
        marker = False
    log_tail = None
    try:
        attrs = sftp.stat(artifact + "/job.log")
        with sftp.file(artifact + "/job.log", "rb") as stream:
            stream.seek(max(0, attrs.st_size - 8192))
            log_tail = stream.read(8192).decode("utf-8", errors="replace")
    except OSError as exc:
        if getattr(exc, "errno", None) != 2:
            raise
    lanes = protocol["jobs"][kind]["lanes"]
    first = lane_observations[lanes[0]] if len(lanes) == 1 else {}
    checkpoints = {f"{lane}/{name}": size
                  for lane, obs in lane_observations.items()
                  for name, size in obs["checkpoint_sizes"].items()}
    return {"job_kind": kind, "lane": lanes[0] if len(lanes) == 1 else None,
            "lane_observations": lane_observations, "result_json": first.get("result_json", {}),
            "result_files": first.get("result_files", {}), "worker_summary": worker_summary,
            "artifact_exit_code": exit_code, "phase_complete": marker,
            "checkpoint_size": sum(size for size in checkpoints.values() if isinstance(size, int)) or None,
            "checkpoint_sizes": checkpoints, "log_tail": log_tail}


def observe(remote, transport, handle: dict, state: dict, kind: str, protocol: dict) -> tuple[dict, dict]:
    job_id = str(handle.get("job_id", ""))
    if not HANDLE.fullmatch(job_id):
        raise ValueError("Invalid PBS job handle")
    current, last_terminal = retain_terminal(scheduler(remote, transport, job_id), state.get("last_terminal"), job_id)
    with remote.sftp_client(transport) as sftp:
        artifacts = artifact_observation(sftp, handle, kind, protocol)
    return {"handle": handle, "scheduler": current, **artifacts}, {**state, "handle": handle,
                                                                     "last_terminal": last_terminal}


def status(remote, transport, kind: str) -> dict:
    protocol = load_protocol()
    with remote.sftp_client(transport) as sftp:
        remote_state = read_remote_json(sftp, remote_state_path(kind)) or {}
        remote_protocol_matches(sftp, protocol)
    state = merged_state(remote_state, read_local_state(kind))
    recovered = recover_intent(remote, transport, state, kind)
    if recovered:
        save_state(remote, transport, kind, state)
    handle = recovered or state.get("handle")
    if not isinstance(handle, dict):
        raise RuntimeError(f"No confirmed PBS handle is available for {kind}")
    assert_handle_scope(handle, kind)
    observation, updated = observe(remote, transport, handle, state, kind, protocol)
    checked = utc_now()
    updated["last_status_utc"] = checked
    save_state(remote, transport, kind, updated)
    return {"checked_utc": checked, **observation}


def query_count(value) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, dict):
        for key in ("completed", "count", "queries"):
            if type(value.get(key)) is int:
                return value[key]
    return None


def finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def metric_values(value) -> list[float] | None:
    if finite_number(value):
        return [float(value)]
    if isinstance(value, dict) and value and all(finite_number(v) for v in value.values()):
        return [float(v) for v in value.values()]
    return None


def expected_exposures(lane: dict) -> int:
    value = lane.get("training_exposures")
    if type(value) is int:
        return value
    return 0


def training_history_failures(training: dict, lane_protocol: dict, kind: str) -> list[str]:
    if training.get("protocol") != PROTOCOL_ID or training.get("lane") != kind:
        return ["training history protocol/lane does not match"]
    planned = lane_protocol.get("arms")
    if not isinstance(planned, list):
        return ["training history has no protocol arm plan"]
    if kind == "path":
        return ([] if training.get("updates") == 0 and training.get("history") == []
                else ["path lane training history must record zero updates"])
    histories = training.get("arms")
    if not isinstance(histories, dict):
        return ["training history arms are missing"]
    planned_names = {item.get("name") for item in planned if isinstance(item, dict)}
    failures = []
    if set(histories) != planned_names:
        failures.append("training history arm set differs from the frozen plan")
    expected_updates = lane_protocol.get("updates", lane_protocol.get("new_training_updates", 0))
    expected_total_exposures = expected_exposures(lane_protocol)
    contexts_per_update = lane_protocol.get("contexts_per_update", 1)
    for name in sorted(planned_names):
        arm = histories.get(name)
        if not isinstance(arm, dict):
            failures.append(f"training history is missing arm {name}")
            continue
        if (arm.get("updates") != expected_updates
                or arm.get("training_exposures") != expected_total_exposures):
            failures.append(f"training history arm {name} update/exposure totals differ from the plan")
        rows = arm.get("history")
        if not isinstance(rows, list) or len(rows) != expected_updates:
            failures.append(f"training history arm {name} must contain {expected_updates} committed updates")
            continue
        row_exposures = 0
        for index, row in enumerate(rows, start=1):
            contexts = row.get("contexts") if isinstance(row, dict) else None
            if (not isinstance(row, dict) or row.get("update") != index
                    or not isinstance(contexts, list) or len(contexts) != contexts_per_update):
                failures.append(f"training history arm {name} update {index} has an invalid cursor or context count")
                continue
            row_exposures += len(contexts)
        if row_exposures != expected_total_exposures:
            failures.append(f"training history arm {name} records {row_exposures} context exposures, expected {expected_total_exposures}")
    return failures


def _lane_verification_failures(observation: dict, protocol: dict, kind: str, lane: str) -> list[str]:
    scheduler_state = observation.get("scheduler") or {}
    handle = observation.get("handle") or {}
    results = observation.get("result_json") or {}
    summary = results.get("summary.json") or {}
    progress = results.get("progress.json") or {}
    selfcheck = results.get("selfcheck.json") or {}
    training = results.get("training.json") or {}
    lane_protocol = protocol["lanes"][lane]
    failures = []
    if scheduler_state.get("job_state") not in ("F", "X"):
        failures.append("PBS job is not in terminal F/X state")
    target_queue = protocol["jobs"][kind]["execution_queue"]
    if scheduler_state.get("queue") != target_queue:
        failures.append(f"PBS job did not route to the verified {target_queue} execution queue")
    if str(scheduler_state.get("Exit_status")) != "0":
        failures.append("PBS Exit_status is not 0")
    if observation.get("artifact_exit_code") != 0:
        failures.append("artifact exit_code.txt is not 0")
    if observation.get("phase_complete") is not True:
        failures.append("PHASE_COMPLETE marker is missing")
    for name in RESULT_JSON_FILES:
        if not isinstance(results.get(name), dict):
            failures.append(f"results/{lane}/{name} is missing or invalid")
    for name in RESULT_FILES:
        if observation.get("result_files", {}).get(name) is not True:
            failures.append(f"results/{lane}/{name} is missing")
    failures.extend(training_history_failures(training, lane_protocol, lane))
    if summary.get("status") != "complete" or summary.get("complete") is not True:
        failures.append("lane summary is not complete")
    reused = observation.get("reuse_verified") is True
    expected_job_id = observation.get("reuse_receipt", {}).get("source_job_id") if reused else handle.get("job_id")
    for key, expected in (("protocol", PROTOCOL_ID), ("lane", lane), ("pbs_jobid", expected_job_id),
                          ("target_linears", 614)):
        if summary.get(key) != expected:
            failures.append(f"lane summary {key} does not match {expected}")
    expected_teachers = 0
    if type(summary.get("new_teacher_queries")) is not int or summary.get("new_teacher_queries") != expected_teachers:
        failures.append(f"lane summary new_teacher_queries is not {expected_teachers}")
    arms = summary.get("arms")
    planned_arms = lane_protocol.get("arms")
    if not isinstance(arms, dict) or not isinstance(planned_arms, list):
        failures.append("lane summary arms or protocol arm plan is missing")
    else:
        expected_updates = lane_protocol.get("updates", lane_protocol.get("new_training_updates", 0))
        exposures = expected_exposures(lane_protocol)
        c80_expected = 80 if lane in ("coverage_c8", "coverage_c80") else 0
        gates = protocol.get("gates", {})
        parity_limit = gates.get("formal_infer_max_abs", 1e-6)
        orthogonality_limit = gates.get("orthogonality_max_abs", 5e-5)
        replay_limit = gates.get("source_replay_metric_max_abs", 2e-6)
        for item in planned_arms:
            name = item.get("name") if isinstance(item, dict) else None
            arm = arms.get(name) if isinstance(name, str) else None
            if not isinstance(arm, dict):
                failures.append(f"lane summary is missing planned arm {name}")
                continue
            for field, expected in (("updates", expected_updates), ("training_exposures", exposures),
                                    ("calibration_queries", 8), ("heldout_queries", 80),
                                    ("c80_queries", c80_expected)):
                actual = query_count(arm.get(field))
                if actual != expected and not (expected == 0 and actual is None):
                    failures.append(f"arm {name} {field} is not {expected}")
            if arm.get("all_parameters_frozen") is not True:
                failures.append(f"arm {name} does not confirm frozen parameters")
            parity = arm.get("formal_infer_max_abs")
            if not finite_number(parity) or parity < 0 or parity > parity_limit:
                failures.append(f"arm {name} formal inference parity exceeds {parity_limit} or is missing")
            orthogonality = metric_values(arm.get("orthogonality"))
            if orthogonality is None or min(orthogonality) < 0 or max(orthogonality) > orthogonality_limit:
                failures.append(f"arm {name} orthogonality exceeds {orthogonality_limit} or is missing")
            movement = metric_values(arm.get("movement"))
            if movement is None or min(movement) < 0:
                failures.append(f"arm {name} movement is missing, negative, or nonfinite")
            elif expected_updates > 0 and max(movement) == 0:
                failures.append(f"trained arm {name} has no measured rotation movement")
            elif isinstance(item, dict) and item.get("t") == 0 and max(movement) > 1e-6:
                failures.append(f"identity path arm {name} moved away from U=I")
            elif isinstance(item, dict) and isinstance(item.get("t"), (int, float)) and item["t"] > 0 and max(movement) == 0:
                failures.append(f"nonidentity path arm {name} has no measured movement")
            source_replay = arm.get("source_replay_metric_max_abs")
            if item.get("reused") is True and (
                not finite_number(source_replay) or source_replay < 0 or source_replay > replay_limit
            ):
                failures.append(f"reused arm {name} source replay exceeds {replay_limit} or is missing")
    if observation.get("worker_summary") != summary:
        failures.append(f"artifacts/{lane}/worker_summary.json differs from the lane summary")
    if observation.get("artifact_exit_code") != 0:
        failures.append(f"lane artifact {lane}/exit_code.txt is not 0")
    if observation.get("phase_complete") is not True:
        failures.append(f"lane artifact {lane}/PHASE_COMPLETE marker is missing")
    if summary.get("pbs_jobid") != handle.get("job_id") and not reused:
        failures.append(f"lane summary {lane} belongs to another job without a verified reuse receipt")
    if summary.get("pbs_jobid") != handle.get("job_id") and reused and not observation.get("reuse_verified"):
        failures.append(f"lane {lane} completed-lane reuse receipt could not be verified")
    if progress.get("protocol") != PROTOCOL_ID or progress.get("lane") != lane:
        failures.append("lane progress protocol/lane does not match")
    if not (progress.get("complete") is True or progress.get("status") == "complete"
            or progress.get("stage") == "complete"):
        failures.append("lane progress does not confirm completion")
    if selfcheck.get("passed") is not True:
        failures.append("lane selfcheck did not pass")
    fixtures = selfcheck.get("fixtures", [])
    if not isinstance(fixtures, list) or not fixtures:
        failures.append("rotation selfcheck fixtures are missing")
        fixtures = []
    for fixture in fixtures:
        if fixture.get("initial_H_A4_exact") is not True or fixture.get("checkpoint_gradient_exact") is not True:
            failures.append("rotation selfcheck lacks exact H/A4 or checkpoint-gradient parity")
            break
    checkpoints = observation.get("checkpoint_sizes") or {}
    if lane == "path":
        if not isinstance(checkpoints.get("state"), int) or checkpoints["state"] <= 0:
            failures.append("path state.pt checkpoint is missing or empty")
    else:
        for item in planned_arms if isinstance(planned_arms, list) else []:
            name = item.get("name") if isinstance(item, dict) else None
            if name and (not isinstance(checkpoints.get(name), int) or checkpoints[name] <= 0):
                failures.append(f"arm {name} resumable snapshot is missing or empty")
    comparison = results.get("comparison.json") or {}
    if (comparison.get("protocol") != PROTOCOL_ID or comparison.get("complete") is not True
            or comparison.get("lane") != lane or comparison.get("target_linears") != 614):
        failures.append("lane comparison report protocol/lane/614-linear completion does not match")
    comparison_splits = [("calibration", 8), ("heldout", 80)]
    if lane in ("coverage_c8", "coverage_c80"):
        comparison_splits.append(("c80", 80))
    for split, expected_queries in comparison_splits:
        split_report = comparison.get(split)
        if not isinstance(split_report, dict):
            failures.append(f"comparison report {split} table is missing")
            continue
        for item in planned_arms if isinstance(planned_arms, list) else []:
            name = item.get("name") if isinstance(item, dict) else None
            row = split_report.get(name) if isinstance(name, str) else None
            if not isinstance(row, dict) or query_count(row.get("queries")) != expected_queries:
                failures.append(f"comparison {split}/{name} does not have {expected_queries} queries")
    return failures


def verification_failures(observation: dict, protocol: dict, kind: str) -> list[str]:
    lanes = protocol["jobs"][kind]["lanes"]
    lane_observations = observation.get("lane_observations")
    failures = []
    scheduler_state = observation.get("scheduler") or {}
    handle = observation.get("handle") or {}
    if scheduler_state.get("job_state") not in ("F", "X"):
        failures.append("PBS job is not in terminal F/X state")
    if str(scheduler_state.get("Exit_status")) != "0":
        failures.append("PBS Exit_status is not 0")
    target_queue = protocol["jobs"][kind]["execution_queue"]
    if scheduler_state.get("queue") != target_queue:
        failures.append(f"PBS job did not route to {target_queue}")
    if observation.get("artifact_exit_code") != 0:
        failures.append("root artifact exit_code.txt is not 0")
    if observation.get("phase_complete") is not True:
        failures.append("root PHASE_COMPLETE marker is missing")
    if not isinstance(lane_observations, dict):
        return failures + ["lane observations are missing"]
    if set(lane_observations) != set(lanes):
        failures.append("lane observations differ from the frozen job plan")
    root_summary = observation.get("worker_summary")
    if len(lanes) == 1:
        lane_summary = lane_observations.get(lanes[0], {}).get("result_json", {}).get("summary.json")
        if root_summary != lane_summary:
            failures.append("artifacts/worker_summary.json differs from the lane summary")
    else:
        if (not isinstance(root_summary, dict) or root_summary.get("protocol") != PROTOCOL_ID
                or root_summary.get("complete") is not True or root_summary.get("status") != "complete"
                or root_summary.get("pbs_jobid") != handle.get("job_id")
                or root_summary.get("job_kind") != kind
                or not isinstance(root_summary.get("lanes"), dict)
                or set(root_summary["lanes"]) != set(lanes)):
            failures.append("root worker_summary.json does not bind the complete multi-lane PBS job")
        else:
            for lane in lanes:
                lane_summary = lane_observations.get(lane, {}).get("result_json", {}).get("summary.json")
                if root_summary["lanes"].get(lane) != lane_summary:
                    failures.append(f"root worker summary differs from results/{lane}/summary.json")
    for lane in lanes:
        lane_observation = lane_observations.get(lane)
        if not isinstance(lane_observation, dict):
            failures.append(f"lane observation {lane} is missing")
            continue
        lane_observation = {**lane_observation, "handle": handle, "scheduler": scheduler_state}
        failures.extend(_lane_verification_failures(lane_observation, protocol, kind, lane))
    return failures


def brief_status(result: dict) -> dict:
    lane_observations = result.get("lane_observations") or {}
    lanes = {}
    for lane, observation in lane_observations.items():
        result_json = observation.get("result_json") or {}
        summary = result_json.get("summary.json") or {}
        progress = result_json.get("progress.json") or {}
        lanes[lane] = {
            "artifact_exit_code": observation.get("artifact_exit_code"),
            "phase_complete": observation.get("phase_complete"),
            "checkpoint_size": observation.get("checkpoint_size"),
            "progress": {key: progress.get(key) for key in ("stage", "complete", "updates", "training_exposures")
                         if key in progress},
            "summary": {key: summary.get(key) for key in (
                "status", "complete", "pbs_jobid", "target_linears", "new_teacher_queries") if key in summary},
        }
    summary = (result.get("result_json") or {}).get("summary.json") or {}
    progress = (result.get("result_json") or {}).get("progress.json") or {}
    return {
        "checked_utc": result.get("checked_utc"), "job_kind": result.get("job_kind"),
        "lane": result.get("lane"), "handle": result.get("handle"), "scheduler": result.get("scheduler"),
        "artifact_exit_code": result.get("artifact_exit_code"), "phase_complete": result.get("phase_complete"),
        "lanes": lanes,
        "checkpoint_size": result.get("checkpoint_size"), "checkpoint_sizes": result.get("checkpoint_sizes"), "progress": {
            key: progress.get(key) for key in ("stage", "complete", "updates", "training_exposures")
            if key in progress
        }, "summary": {
            key: summary.get(key) for key in ("status", "complete", "lane", "target_linears", "new_teacher_queries")
            if key in summary
        }, "arms": {
            name: {key: arm.get(key) for key in (
                "updates", "training_exposures", "calibration_queries", "heldout_queries",
                "c80_queries", "formal_infer_max_abs", "source_replay_metric_max_abs", "orthogonality", "movement",
            ) if key in arm}
            for name, arm in (summary.get("arms") or {}).items()
        }, "log_tail": result.get("log_tail"),
    }


def verify(remote, transport, kind: str) -> dict:
    protocol = load_protocol()
    result = status(remote, transport, kind)
    failures = verification_failures(result, protocol, kind)
    checked = utc_now()
    state = merged_state({}, read_local_state(kind))
    state.update(handle=result["handle"], last_status_utc=checked)
    if not failures:
        state["last_verified"] = {"job_id": result["handle"]["job_id"], "verified_utc": checked,
                                  "job_kind": kind, "summary": (result.get("result_json") or {}).get("summary.json")}
    save_state(remote, transport, kind, state)
    return {"checked_utc": checked, "verified": not failures, "failures": failures,
            **brief_status(result)}


def resume_failures(observation: dict, protocol: dict, kind: str) -> list[str]:
    scheduler_state = observation.get("scheduler") or {}
    handle = observation.get("handle") or {}
    confirmed_terminal = scheduler_state.get("job_state") in ("F", "X") and (
        scheduler_state.get("query_rc") == 0 or (
            scheduler_state.get("terminal_from_history") is True and scheduler_state.get("job_id") == handle.get("job_id")
        )
    )
    failures = []
    if not confirmed_terminal:
        failures.append("current PBS handle is not confirmed terminal F/X; query timeout never authorizes resume")
    if scheduler_state.get("Exit_status") is None:
        failures.append("PBS terminal receipt has no Exit_status")
    if str(scheduler_state.get("Exit_status")) == "0":
        failures.append("PBS Exit_status is 0; refusing to resume a successful job")
    if observation.get("artifact_exit_code") in (None, 0):
        failures.append("a nonzero artifacts/exit_code.txt receipt is required to resume")
    if observation.get("phase_complete") is True:
        failures.append("PHASE_COMPLETE exists; refusing to resume a completed PBS job")
    if protocol.get("protocol") != PROTOCOL_ID:
        failures.append("local protocol does not match the frozen protocol ID")
    lane_observations = observation.get("lane_observations") or {}
    incomplete = []
    for lane in protocol["jobs"][kind]["lanes"]:
        lane_observation = lane_observations.get(lane)
        if not isinstance(lane_observation, dict):
            failures.append(f"lane {lane} has no observation; safe resume is unavailable")
            continue
        lane_results = lane_observation.get("result_json") or {}
        summary = lane_results.get("summary.json") or {}
        progress = lane_results.get("progress.json") or {}
        complete_summary = summary.get("status") == "complete" and summary.get("complete") is True
        receipt_valid = (lane_observation.get("artifact_exit_code") == 0
                         and lane_observation.get("phase_complete") is True
                         and lane_observation.get("worker_summary") == summary
                         and (summary.get("pbs_jobid") == handle.get("job_id")
                              or lane_observation.get("reuse_verified") is True))
        if complete_summary:
            if not receipt_valid:
                failures.append(f"lane {lane} claims completion without a valid immutable lane receipt")
            continue
        incomplete.append(lane)
        if progress.get("complete") is True or progress.get("status") == "complete":
            failures.append(f"lane {lane} progress claims completion but its summary/receipt is incomplete")
        if not isinstance(progress, dict) or progress.get("protocol") != PROTOCOL_ID or progress.get("lane") != lane:
            failures.append(f"lane {lane} progress does not identify the frozen protocol")
        checkpoints = lane_observation.get("checkpoint_sizes") or {}
        if not any(type(size) is int and size > 0 for size in checkpoints.values()):
            failures.append(f"lane {lane} has no nonempty checkpoint; safe resume is unavailable")
    if not incomplete:
        failures.append("all lanes already claim completion or no incomplete resumable lane exists")
    return failures


def resume(remote, transport, kind: str) -> dict:
    protocol = load_protocol()
    with remote.sftp_client(transport) as sftp:
        remote_state = read_remote_json(sftp, remote_state_path(kind)) or {}
        remote_protocol_matches(sftp, protocol)
    state = merged_state(remote_state, read_local_state(kind))
    recovered = recover_intent(remote, transport, state, kind)
    if recovered:
        save_state(remote, transport, kind, state)
        return {"recovered": True, "job_kind": kind, "handle": recovered}
    handle = state.get("handle")
    if not isinstance(handle, dict) or not HANDLE.fullmatch(str(handle.get("job_id", ""))):
        raise RuntimeError("Resume requires a confirmed current PBS handle")
    assert_handle_scope(handle, kind)
    observation, state = observe(remote, transport, handle, state, kind, protocol)
    state["last_status_utc"] = utc_now()
    save_state(remote, transport, kind, state)
    failures = resume_failures(observation, protocol, kind)
    if failures:
        raise RuntimeError("Resume refused: " + "; ".join(failures))
    with remote.sftp_client(transport) as sftp:
        assert_remote_files(sftp)
    attempt = attempt_number(handle, kind) + 1
    name = job_name(kind, attempt)
    if named_jobs(remote, transport, name):
        raise RuntimeError(f"Candidate job name {name} already exists in PBS history; refusing duplicate submission")
    target_queue, select, walltime = job_resources(kind, protocol)
    intent = {"kind": "resume", "job_kind": kind, "status": "intent_before_qsub", "attempt": attempt,
              "job_name": name, "from_job_id": handle["job_id"], "script": ROOT + "/run.pbs",
              "created_utc": utc_now(), "queue": SUBMISSION_QUEUE,
              "target_queue": target_queue, "project": PROJECT,
              "select": select, "walltime": walltime}
    state["intent"] = intent
    save_state(remote, transport, kind, state)
    try:
        code, output = remote.command(transport, qsub_command(kind, name, attempt, handle["job_id"]), timeout=40)
    except Exception as exc:
        intent.update(status="qsub_outcome_unknown", error=type(exc).__name__, updated_utc=utc_now())
        save_state(remote, transport, kind, state)
        raise RuntimeError("Resume qsub outcome is unknown; a later resume may only recover its exact PBS job") from exc
    job_id = output.strip()
    if code or not HANDLE.fullmatch(job_id):
        intent.update(status="qsub_outcome_unknown", response=output[:1000], return_code=code, updated_utc=utc_now())
        save_state(remote, transport, kind, state)
        raise RuntimeError("Resume qsub outcome is unknown or rejected; do not submit another attempt")
    new_handle = {"job_id": job_id, "job_name": name, "attempt": attempt, "remote_root": ROOT,
                  "protocol": PROTOCOL_ID, "job_kind": kind, "status": "confirmed", "confirmed_utc": utc_now()}
    intent.update(status="confirmed", job_id=job_id, confirmed_utc=utc_now())
    state.update(intent=intent, previous_handle=handle, handle=new_handle)
    save_state(remote, transport, kind, state)
    return {"resumed": True, "job_kind": kind, "from_job_id": handle["job_id"],
            "handle": new_handle, "resume_mode": "checkpoint"}


def pull(remote, transport, kind: str) -> dict:
    protocol = load_protocol()
    result = status(remote, transport, kind)
    job_id = result["handle"]["job_id"]
    lanes = protocol["jobs"][kind]["lanes"]
    destination = HERE / "results" / (lanes[0] if len(lanes) == 1 else kind) / job_id
    destination.mkdir(parents=True, exist_ok=True)
    lane_directories = {}
    with remote.sftp_client(transport) as sftp:
        artifact = ROOT + "/artifacts/" + job_id
        for name in ("worker_summary.json", "exit_code.txt"):
            data = read_small_bytes(sftp, artifact + "/" + name)
            if data is not None:
                (destination / name).write_bytes(data)
        try:
            sftp.stat(artifact + "/PHASE_COMPLETE")
            (destination / "PHASE_COMPLETE").write_text("\n", encoding="ascii")
        except OSError as exc:
            if getattr(exc, "errno", None) != 2:
                raise
        for lane in lanes:
            lane_destination = destination if len(lanes) == 1 else destination / lane
            lane_destination.mkdir(parents=True, exist_ok=True)
            lane_directories[lane] = str(lane_destination)
            for name in RESULT_FILES:
                data = read_small_bytes(sftp, ROOT + "/results/" + lane + "/" + name)
                if data is not None:
                    (lane_destination / name).write_bytes(data)
            lane_artifact = artifact if len(lanes) == 1 else artifact + "/" + lane
            for name in ("worker_summary.json", "exit_code.txt", "reuse_receipt.json"):
                data = read_small_bytes(sftp, lane_artifact + "/" + name)
                if data is not None:
                    (lane_destination / name).write_bytes(data)
            try:
                sftp.stat(lane_artifact + "/PHASE_COMPLETE")
                (lane_destination / "PHASE_COMPLETE").write_text("\n", encoding="ascii")
            except OSError as exc:
                if getattr(exc, "errno", None) != 2:
                    raise
            lane_result = result.get("lane_observations", {}).get(lane, {})
            if lane_result.get("log_tail") is not None:
                (lane_destination / "job.log.tail").write_text(lane_result["log_tail"], encoding="utf-8")
    metadata = {key: result.get(key) for key in (
        "checked_utc", "handle", "scheduler", "artifact_exit_code", "phase_complete", "checkpoint_size",
        "checkpoint_sizes", "lane", "lane_observations",
    )}
    save_local(destination / "pull_metadata.json", metadata)
    return {"job_kind": kind, "job_id": job_id, "local_directory": str(destination),
            "lane_directories": lane_directories, "metadata": brief_status(result)}


def stage_action(remote, transport, kind: str) -> dict:
    return stage(remote, transport, kind)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "submit", "status", "verify", "pull", "resume"))
    parser.add_argument("--job-kind", choices=KINDS, required=True)
    args = parser.parse_args()
    helper = load_remote_helper()
    jump, transport = helper.connect()
    try:
        if args.action == "stage":
            result = stage_action(helper, transport, args.job_kind)
        elif args.action == "submit":
            result = submit(helper, transport, args.job_kind)
        elif args.action == "status":
            result = brief_status(status(helper, transport, args.job_kind))
        elif args.action == "verify":
            result = verify(helper, transport, args.job_kind)
        elif args.action == "resume":
            result = resume(helper, transport, args.job_kind)
        else:
            result = pull(helper, transport, args.job_kind)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0 if result.get("verified", True) else 1
    finally:
        transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
