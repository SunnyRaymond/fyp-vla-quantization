from __future__ import annotations

import re
import socket
import sys
import time

import paramiko

from nscc_jump_probe import (
    JUMP_KNOWN_HOSTS,
    NSCC_KNOWN_HOSTS,
    read_credentials,
    verify_host_key,
)


TARGET_HOST = "aspire2antu.nscc.sg"
PROJECT_ID = "personal-yguo017"


def run(transport: paramiko.Transport, command: str, timeout: int = 60):
    channel = transport.open_session(timeout=20)
    channel.settimeout(timeout)
    channel.exec_command(command)
    stdout = channel.makefile("rb").read().decode("utf-8", errors="replace")
    stderr = channel.makefile_stderr("rb").read().decode(
        "utf-8", errors="replace"
    )
    status = channel.recv_exit_status()
    return status, stdout.strip(), stderr.strip()


def connect():
    credentials = read_credentials()
    jump_host = credentials["NTU_HOST"]

    jump_socket = socket.create_connection((jump_host, 22), timeout=20)
    jump = paramiko.Transport(jump_socket)
    jump.start_client(timeout=20)
    verify_host_key(jump, jump_host, JUMP_KNOWN_HOSTS)

    def ntu_password_handler(title, instructions, prompts):
        return [credentials["NTU_PASSWORD"] for _prompt, _echo in prompts]

    jump.auth_interactive(credentials["NTU_USERNAME"], ntu_password_handler)
    if not jump.is_authenticated():
        raise RuntimeError("NTU Jump Host authentication failed.")

    channel = jump.open_channel(
        "direct-tcpip", (TARGET_HOST, 22), ("127.0.0.1", 0), timeout=20
    )
    nscc = paramiko.Transport(channel)
    nscc.start_client(timeout=20)
    verify_host_key(nscc, TARGET_HOST, NSCC_KNOWN_HOSTS)
    nscc.auth_password(
        credentials["NSCC_USERNAME"], credentials["NSCC_PASSWORD"]
    )
    if not nscc.is_authenticated():
        raise RuntimeError("ASPIRE2A authentication failed.")
    return jump, nscc


def main() -> int:
    jump, nscc = connect()
    status, home, error = run(nscc, "printf %s \"$HOME\"")
    if status or not home.startswith("/"):
        raise RuntimeError(f"Could not determine remote home directory: {error}")

    output_path = f"{home}/codex-node-probe.out"
    script = r"""#!/bin/bash
#PBS -N codex_node_probe
#PBS -q normal
#PBS -P personal-yguo017
#PBS -l select=1:ncpus=16:mem=124gb:ngpus=1
#PBS -l walltime=00:05:00
#PBS -j oe

printf '=== ALLOCATED NODE ===\n'
hostname -f
date
printf '\n=== OPERATING SYSTEM ===\n'
sed -n '1,8p' /etc/os-release
uname -srmo
printf '\n=== CPU ===\n'
lscpu
printf '\n=== MEMORY ===\n'
free -h
printf '\n=== GPU ===\n'
nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id --format=csv
printf '\n=== LOCAL STORAGE ===\n'
lsblk -o NAME,MODEL,SIZE,TYPE,MOUNTPOINTS
printf '\n=== FILESYSTEMS ===\n'
df -hT "$HOME" /scratch 2>&1
printf '\n=== PBS ALLOCATION ===\n'
printf 'PBS_JOBID=%s\nPBS_QUEUE=%s\nPBS_NODEFILE=%s\n' "$PBS_JOBID" "$PBS_QUEUE" "$PBS_NODEFILE"
cat "$PBS_NODEFILE"
"""

    submit = nscc.open_session(timeout=20)
    submit.settimeout(60)
    command = (
        f"qsub -o {output_path} "
        "-N codex_node_probe -q normal -P personal-yguo017 "
        "-l select=1:ncpus=16:mem=124gb:ngpus=1 "
        "-l walltime=00:05:00 -j oe"
    )
    submit.exec_command(command)
    submit.sendall(script.encode("utf-8"))
    submit.shutdown_write()
    job_id = submit.makefile("rb").read().decode("utf-8", errors="replace").strip()
    submit_error = submit.makefile_stderr("rb").read().decode(
        "utf-8", errors="replace"
    ).strip()
    submit_status = submit.recv_exit_status()
    if submit_status or not re.fullmatch(r"\d+(?:\.[A-Za-z0-9._-]+)?", job_id):
        raise RuntimeError(f"PBS submission failed: {submit_error or job_id}")
    print(f"Submitted PBS job: {job_id}", flush=True)
    print(f"Output path: {output_path}", flush=True)

    final_state = ""
    for _ in range(72):
        status, output, error = run(
            nscc,
            f"qstat -f {job_id} 2>/dev/null | "
            "awk -F ' = ' '/job_state =|queue =|exec_host =|comment =/ {print $1 \"=\" $2}'",
        )
        if output:
            state_match = re.search(r"job_state\s*=\s*([A-Z])", output)
            state = state_match.group(1) if state_match else "?"
            if state != final_state:
                print(f"PBS state: {state}", flush=True)
                print(output, flush=True)
                final_state = state
            if state == "F":
                break
        if status != 0 or not output:
            break
        time.sleep(5)

    for _ in range(12):
        status, output, error = run(
            nscc,
            f"if test -s {output_path}; then cat {output_path}; else exit 1; fi",
        )
        if status == 0:
            print("\n=== COMPUTE NODE REPORT ===", flush=True)
            print(output, flush=True)
            break
        time.sleep(2)
    else:
        print("The PBS job has not produced its report yet.", flush=True)

    nscc.close()
    jump.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Compute probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
