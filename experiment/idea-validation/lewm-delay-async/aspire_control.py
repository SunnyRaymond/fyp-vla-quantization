"""Small ASPIRE2A control-file transfers and PBS status; no login-node workload."""

from __future__ import annotations

import argparse
import importlib.util
import re
import shlex
from pathlib import Path, PurePosixPath

import paramiko


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
REMOTE = "/scratch/users/ntu/yguo017/dino-wm-wall/experiment/idea-validation/lewm-delay-async"
MAX_CONTROL_BYTES = 512_000


def connection():
    helper = PROJECT / "nscc-access" / "aspire2a_shell.py"
    spec = importlib.util.spec_from_file_location("aspire2a_shell", helper)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.connect()


def run(transport, command: str) -> str:
    channel = transport.open_session(timeout=30)
    channel.exec_command(command)
    stdout = channel.makefile("rb").read(MAX_CONTROL_BYTES + 1)
    stderr = channel.makefile_stderr("rb").read(MAX_CONTROL_BYTES + 1)
    status = channel.recv_exit_status()
    if len(stdout) > MAX_CONTROL_BYTES or len(stderr) > MAX_CONTROL_BYTES:
        raise RuntimeError("Control output exceeded size limit")
    if status:
        raise RuntimeError(f"Remote command failed ({status}): {stderr.decode(errors='replace')}")
    return stdout.decode(errors="replace")


def clean_relative(value: str) -> Path:
    rel = Path(value)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts:
        raise ValueError("Expected a relative path inside this experiment")
    return rel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "queue", "upload", "submit", "status", "read"))
    parser.add_argument("value", nargs="?")
    parser.add_argument("local", nargs="?")
    args = parser.parse_args()
    jump = nscc = None
    try:
        jump, nscc = connection()
        if args.action == "check":
            print(run(nscc, "hostname; pwd; ls -ld /scratch/users/ntu/yguo017 /scratch/users/ntu/yguo017/dino-wm-wall " + shlex.quote(REMOTE) + " " + shlex.quote(REMOTE + "/reacher") + " 2>&1 || true; qstat -u $(id -un) -answ1 || true"), end="")
        elif args.action == "queue":
            print(run(nscc, "qstat -Qf gdev"), end="")
        elif args.action == "upload":
            rel = clean_relative(args.value or "")
            source = (HERE / rel).resolve(strict=True)
            if not source.is_relative_to(HERE) or source.suffix not in (".py", ".pbs", ".json", ".md", ".sh"):
                raise ValueError("Only experiment control files may be uploaded")
            if re.search("credential|password|secret|private", source.name, re.I):
                raise ValueError("Refusing a sensitive file name")
            content = source.read_bytes().replace(b"\r\n", b"\n")
            if len(content) > MAX_CONTROL_BYTES:
                raise ValueError("Control file exceeds size limit")
            destination = REMOTE + "/" + rel.as_posix()
            print(run(nscc, "mkdir -p " + shlex.quote(str(PurePosixPath(destination).parent))))
            sftp = paramiko.SFTPClient.from_transport(nscc)
            try:
                with sftp.open(destination + ".uploading", "wb") as stream:
                    stream.write(content)
                sftp.posix_rename(destination + ".uploading", destination)
            finally:
                sftp.close()
            print(f"uploaded {rel.as_posix()} ({len(content)} bytes)")
        elif args.action == "submit":
            rel = clean_relative(args.value or "")
            if rel.suffix != ".pbs":
                raise ValueError("Expected a PBS control file")
            allowed_variables = {
                "LEWM_MODE", "LEWM_DELAY_STEPS", "LEWM_TASK_START", "LEWM_TASK_STOP",
                "LEWM_TASK_COUNT", "LEWM_GATE_DIR", "LEWM_BATCH50_GATE",
                "LEWM_N1_K0_GATES", "LEWM_MANIFEST",
                "LEWM_AGGREGATION_MAP", "LEWM_AGGREGATION_OUTPUT",
            }
            variables = []
            if args.local:
                for assignment in args.local.split(","):
                    if "=" not in assignment:
                        raise ValueError("Expected comma-separated PBS NAME=value variables")
                    name, value = assignment.split("=", 1)
                    if name not in allowed_variables or not re.fullmatch(r"[A-Za-z0-9_./:-]+", value):
                        raise ValueError("Invalid or disallowed PBS variable")
                    variables.append(f"{name}={value}")
            variable_arg = " -v " + shlex.quote(",".join(variables)) if variables else ""
            print(run(nscc, "qsub" + variable_arg + " " + shlex.quote(REMOTE + "/" + rel.as_posix())), end="")
        elif args.action == "status":
            job = args.value or ""
            if not re.fullmatch(r"[0-9]+(?:\.[A-Za-z0-9.-]+)?", job):
                raise ValueError("Invalid PBS job ID")
            print(run(nscc, "qstat -x -f " + shlex.quote(job)), end="")
        elif args.action == "read":
            rel = clean_relative(args.value or "")
            if rel.name != "job_status" and rel.suffix not in (".json", ".jsonl", ".log", ".txt", ".out", ".err"):
                raise ValueError("Expected a small result/log file")
            sftp = paramiko.SFTPClient.from_transport(nscc)
            try:
                target = REMOTE + "/" + rel.as_posix()
                if sftp.stat(target).st_size > MAX_CONTROL_BYTES:
                    raise ValueError("Result is too large for a login-node read")
                with sftp.open(target, "rb") as stream:
                    content = stream.read(MAX_CONTROL_BYTES + 1)
            finally:
                sftp.close()
            if len(content) > MAX_CONTROL_BYTES:
                raise ValueError("Result exceeded size limit")
            if args.local:
                local_rel = clean_relative(args.local)
                local = HERE / local_rel
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(content)
            else:
                print(content.decode(errors="replace"), end="")
    finally:
        if nscc is not None:
            nscc.close()
        if jump is not None:
            jump.close()


if __name__ == "__main__":
    main()
