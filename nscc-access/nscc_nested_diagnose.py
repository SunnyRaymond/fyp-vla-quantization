from __future__ import annotations

import socket
import sys
import time
import re

import paramiko

from nscc_jump_probe import (
    JUMP_KNOWN_HOSTS,
    read_credentials,
    verify_host_key,
)


def main() -> int:
    credentials = read_credentials()
    jump_host = credentials["NTU_HOST"]
    target_host = (
        sys.argv[1] if len(sys.argv) > 1 else credentials["NSCC_HOST"]
    )
    if not re.fullmatch(r"[A-Za-z0-9.-]+", target_host):
        raise RuntimeError("Invalid target host.")

    jump_socket = socket.create_connection((jump_host, 22), timeout=20)
    transport = paramiko.Transport(jump_socket)
    transport.start_client(timeout=20)
    verify_host_key(transport, jump_host, JUMP_KNOWN_HOSTS)

    def password_handler(title, instructions, prompts):
        return [credentials["NTU_PASSWORD"] for _prompt, _echo in prompts]

    transport.auth_interactive(credentials["NTU_USERNAME"], password_handler)
    if not transport.is_authenticated():
        raise RuntimeError("NTU Jump Host authentication failed.")

    nscc_target = f"{credentials['NSCC_USERNAME']}@{target_host}"
    remote_command = (
        "ssh -v -T -o ConnectTimeout=20 -o NumberOfPasswordPrompts=1 "
        "-o PubkeyAuthentication=no -o PreferredAuthentications=password "
        f"-o StrictHostKeyChecking=accept-new {nscc_target} "
        "\"printf '__NSCC_CONNECTED__\\\\n'; hostname; id -un\""
    )

    channel = transport.open_session(timeout=20)
    channel.get_pty(term="xterm", width=120, height=40)
    channel.exec_command(remote_command)

    output = ""
    password_sent = False
    deadline = time.monotonic() + 35
    while time.monotonic() < deadline:
        if channel.recv_ready():
            chunk = channel.recv(65535).decode("utf-8", errors="replace")
            output += chunk
            if not password_sent and "password:" in output.lower():
                channel.send(credentials["NSCC_PASSWORD"] + "\n")
                password_sent = True
        if channel.recv_stderr_ready():
            output += channel.recv_stderr(65535).decode(
                "utf-8", errors="replace"
            )
        if channel.exit_status_ready() and not channel.recv_ready():
            break
        time.sleep(0.1)

    if not channel.exit_status_ready():
        channel.close()
        output += "\nDiagnostic timed out.\n"
        status = 124
    else:
        status = channel.recv_exit_status()

    transport.close()
    output = output.replace(credentials["NSCC_PASSWORD"], "[REDACTED]")
    print(output)
    return status


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Diagnostic failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
