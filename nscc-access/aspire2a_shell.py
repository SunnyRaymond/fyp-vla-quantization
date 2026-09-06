from __future__ import annotations

import argparse
import msvcrt
import shutil
import socket
import sys
import threading
from pathlib import Path

import paramiko


ACCESS_DIR = Path(__file__).resolve().parent
CREDENTIAL_PATH = ACCESS_DIR / "nscc-credentials.env"
ASPIRE2A_HOST = "aspire2antu.nscc.sg"
JUMP_KNOWN_HOSTS = Path.home() / ".ssh" / "known_hosts_ntu_jump"
NSCC_KNOWN_HOSTS = Path.home() / ".ssh" / "known_hosts_nscc_aspire2a"


def read_credentials() -> dict[str, str]:
    if not CREDENTIAL_PATH.is_file():
        raise RuntimeError(f"Credential file not found: {CREDENTIAL_PATH}")

    fields: dict[str, str] = {}
    for line in CREDENTIAL_PATH.read_text(encoding="utf-8-sig").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()

    required = ("NTU_HOST", "NTU_USERNAME", "NTU_PASSWORD", "NSCC_USERNAME", "NSCC_PASSWORD")
    missing = [name for name in required if not fields.get(name)]
    if missing:
        raise RuntimeError(f"Missing credential fields: {', '.join(missing)}")
    return fields


def verify_host_key(transport: paramiko.Transport, host: str, known_hosts_path: Path) -> None:
    if not known_hosts_path.is_file():
        raise RuntimeError(f"Trusted host-key file not found: {known_hosts_path}")
    known_hosts = paramiko.HostKeys(str(known_hosts_path))
    expected = known_hosts.lookup(host)
    received = transport.get_remote_server_key()
    if not expected or received.get_name() not in expected:
        raise RuntimeError(f"No trusted {received.get_name()} key is recorded for {host}.")
    if expected[received.get_name()] != received:
        raise RuntimeError(f"Host-key mismatch for {host}; connection stopped for safety.")


def connect() -> tuple[paramiko.Transport, paramiko.Transport]:
    credentials = read_credentials()
    jump_host = credentials["NTU_HOST"]

    jump_socket = socket.create_connection((jump_host, 22), timeout=20)
    jump_transport = paramiko.Transport(jump_socket)
    jump_transport.start_client(timeout=20)
    verify_host_key(jump_transport, jump_host, JUMP_KNOWN_HOSTS)

    def ntu_password_handler(_title, _instructions, prompts):
        return [credentials["NTU_PASSWORD"] for _prompt, _echo in prompts]

    jump_transport.auth_interactive(credentials["NTU_USERNAME"], ntu_password_handler)
    if not jump_transport.is_authenticated():
        raise RuntimeError("NTU Jump Host authentication failed.")

    tunnel = jump_transport.open_channel(
        "direct-tcpip", (ASPIRE2A_HOST, 22), ("127.0.0.1", 0), timeout=20
    )
    nscc_transport = paramiko.Transport(tunnel)
    nscc_transport.start_client(timeout=20)
    verify_host_key(nscc_transport, ASPIRE2A_HOST, NSCC_KNOWN_HOSTS)
    nscc_transport.auth_password(credentials["NSCC_USERNAME"], credentials["NSCC_PASSWORD"])
    if not nscc_transport.is_authenticated():
        raise RuntimeError("ASPIRE2A authentication failed.")

    return jump_transport, nscc_transport


def run_check(nscc_transport: paramiko.Transport) -> int:
    session = nscc_transport.open_session(timeout=20)
    session.exec_command("printf 'Connected to '; hostname; printf 'User: '; id -un")
    output = session.makefile("rb").read().decode("utf-8", errors="replace")
    error = session.makefile_stderr("rb").read().decode("utf-8", errors="replace")
    status = session.recv_exit_status()
    print(output, end="" if output.endswith("\n") else "\n")
    if error:
        print(error, file=sys.stderr, end="" if error.endswith("\n") else "\n")
    return status


def keyboard_input(channel: paramiko.Channel) -> None:
    special_keys = {
        "H": "\x1b[A",  # Up
        "P": "\x1b[B",  # Down
        "M": "\x1b[C",  # Right
        "K": "\x1b[D",  # Left
        "G": "\x1b[H",  # Home
        "O": "\x1b[F",  # End
        "S": "\x1b[3~",  # Delete
        "I": "\x1b[5~",  # Page Up
        "Q": "\x1b[6~",  # Page Down
    }
    try:
        while not channel.closed:
            char = msvcrt.getwch()
            if char in ("\x00", "\xe0"):
                sequence = special_keys.get(msvcrt.getwch())
                if sequence:
                    channel.send(sequence)
            elif char == "\r":
                channel.send("\r")
            elif char == "\x08":
                channel.send("\x7f")
            else:
                channel.send(char.encode("utf-8"))
    except (EOFError, OSError, paramiko.SSHException):
        pass


def interactive_shell(nscc_transport: paramiko.Transport) -> int:
    columns, rows = shutil.get_terminal_size(fallback=(120, 30))
    channel = nscc_transport.open_session(timeout=20)
    channel.get_pty(term="xterm-256color", width=columns, height=rows)
    channel.invoke_shell()

    print("Connected securely through the NTU Jump Host. Type 'exit' to disconnect.\n")
    input_thread = threading.Thread(target=keyboard_input, args=(channel,), daemon=True)
    input_thread.start()

    while True:
        data = channel.recv(32768)
        if not data:
            break
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    return channel.recv_exit_status()


def main() -> int:
    parser = argparse.ArgumentParser(description="Connect to NSCC ASPIRE2A through the NTU Jump Host.")
    parser.add_argument("--check", action="store_true", help="Verify login, print the remote host, and exit.")
    args = parser.parse_args()

    jump_transport: paramiko.Transport | None = None
    nscc_transport: paramiko.Transport | None = None
    try:
        jump_transport, nscc_transport = connect()
        return run_check(nscc_transport) if args.check else interactive_shell(nscc_transport)
    finally:
        if nscc_transport is not None:
            nscc_transport.close()
        if jump_transport is not None:
            jump_transport.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nConnection closed.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"Connection failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
