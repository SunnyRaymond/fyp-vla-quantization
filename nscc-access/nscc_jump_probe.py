from __future__ import annotations

import socket
import sys
import re
from pathlib import Path

import paramiko


ACCESS_DIR = Path(__file__).resolve().parent
CREDENTIAL_PATH = ACCESS_DIR / "nscc-credentials.env"
JUMP_KNOWN_HOSTS = Path(r"C:\Users\Raymond\.ssh\known_hosts_ntu_jump")
NSCC_KNOWN_HOSTS = Path(r"C:\Users\Raymond\.ssh\known_hosts_nscc_aspire2a")


def read_credentials() -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in CREDENTIAL_PATH.read_text(encoding="utf-8-sig").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value
    required = {
        "NTU_HOST",
        "NTU_USERNAME",
        "NTU_PASSWORD",
        "NSCC_HOST",
        "NSCC_USERNAME",
        "NSCC_PASSWORD",
    }
    missing = sorted(name for name in required if not fields.get(name))
    if missing:
        raise RuntimeError(f"Missing credential fields: {', '.join(missing)}")
    return fields


def verify_host_key(
    transport: paramiko.Transport, host: str, known_hosts_path: Path
) -> None:
    known_hosts = paramiko.HostKeys(str(known_hosts_path))
    expected = known_hosts.lookup(host)
    received = transport.get_remote_server_key()
    if not expected or received.get_name() not in expected:
        raise RuntimeError(f"No trusted {received.get_name()} key recorded for {host}.")
    if expected[received.get_name()] != received:
        raise RuntimeError(f"Host key mismatch for {host}.")


def main() -> int:
    credentials = read_credentials()
    jump_host = credentials["NTU_HOST"]
    nscc_host = sys.argv[1] if len(sys.argv) > 1 else credentials["NSCC_HOST"]
    if not re.fullmatch(r"[A-Za-z0-9.-]+", nscc_host):
        raise RuntimeError("Invalid NSCC target host.")

    jump_socket = socket.create_connection((jump_host, 22), timeout=20)
    jump_transport = paramiko.Transport(jump_socket)
    jump_transport.start_client(timeout=20)
    verify_host_key(jump_transport, jump_host, JUMP_KNOWN_HOSTS)

    def ntu_password_handler(title, instructions, prompts):
        return [credentials["NTU_PASSWORD"] for _prompt, _echo in prompts]

    jump_transport.auth_interactive(
        credentials["NTU_USERNAME"], ntu_password_handler
    )
    if not jump_transport.is_authenticated():
        raise RuntimeError("NTU Jump Host authentication failed.")
    print("NTU Jump Host authentication: OK")

    channel = jump_transport.open_channel(
        "direct-tcpip", (nscc_host, 22), ("127.0.0.1", 0), timeout=20
    )
    nscc_transport = paramiko.Transport(channel)
    nscc_transport.start_client(timeout=20)
    verify_host_key(nscc_transport, nscc_host, NSCC_KNOWN_HOSTS)
    nscc_transport.auth_password(
        credentials["NSCC_USERNAME"], credentials["NSCC_PASSWORD"]
    )
    if not nscc_transport.is_authenticated():
        raise RuntimeError("NSCC authentication failed.")
    print(f"NSCC authentication to {nscc_host}: OK")

    command = r"""
printf '\n=== LOGIN NODE ===\n'
hostname
date
uname -srmo
printf '\n--- OS ---\n'
sed -n '1,8p' /etc/os-release
printf '\n--- CPU ---\n'
lscpu | grep -E '^(Architecture|CPU\(s\)|On-line CPU|Thread|Core|Socket|Model name|NUMA node\(s\))'
printf '\n--- MEMORY ---\n'
free -h
printf '\n=== PROJECT ALLOCATION ===\n'
myprojects
printf '\n=== PERSONAL USAGE ===\n'
myusage
printf '\n=== STORAGE QUOTA ===\n'
myquota
printf '\n=== PBS QUEUES ===\n'
qstat -Q
printf '\n=== SOFTWARE MODULE SAMPLE ===\n'
module avail 2>&1 | head -80
""".strip()

    session = nscc_transport.open_session(timeout=20)
    session.settimeout(60)
    session.exec_command(command)
    stdout = session.makefile("rb").read().decode("utf-8", errors="replace")
    stderr = session.makefile_stderr("rb").read().decode(
        "utf-8", errors="replace"
    )
    exit_status = session.recv_exit_status()
    print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print("\n=== STDERR ===", file=sys.stderr)
        print(stderr, file=sys.stderr)

    nscc_transport.close()
    jump_transport.close()
    return exit_status


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
