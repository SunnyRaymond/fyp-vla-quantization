"""SSH helper for NTU EEE Cluster 02 (Slurm).

Usage:
  python -Xutf8 -B eee-access/eee_shell.py run "<command>"
  python -Xutf8 -B eee-access/eee_shell.py first-login   # temp -> new password

Credentials come from the root credentials.env (EEE_HOST, EEE_USERNAME,
EEE_PASSWORD, EEE_temp_password). The host key is pinned to the fingerprint
published in the cluster quickstart.
"""

from __future__ import annotations

import base64
import hashlib
import socket
import sys
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
HOST_KEY_SHA256 = "AhYlWQBTsN/4GAzYTVTiZzrVhPhcurFMu1sBMglqvdM"


def read_credentials() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in (ROOT / "credentials.env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith(("#", "@")):
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def open_transport(host: str) -> paramiko.Transport:
    transport = paramiko.Transport(socket.create_connection((host, 22), timeout=20))
    transport.start_client(timeout=20)
    key = transport.get_remote_server_key()
    fingerprint = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")
    if fingerprint != HOST_KEY_SHA256:
        transport.close()
        raise RuntimeError(f"EEE host key mismatch: SHA256:{fingerprint}")
    return transport


def connect() -> paramiko.Transport:
    env = read_credentials()
    transport = open_transport(env["EEE_HOST"])

    def handler(_title, _instructions, prompts):
        return [env["EEE_PASSWORD"] for _prompt, _echo in prompts]

    transport.auth_interactive(env["EEE_USERNAME"], handler)
    if not transport.is_authenticated():
        raise RuntimeError("EEE authentication failed.")
    return transport


def run(transport: paramiko.Transport, command: str, timeout: float = 120) -> int:
    session = transport.open_session(timeout=20)
    session.settimeout(timeout)
    session.exec_command(f"bash -lc {quote(command)}")
    out = session.makefile("rb").read().decode("utf-8", errors="replace")
    err = session.makefile_stderr("rb").read().decode("utf-8", errors="replace")
    status = session.recv_exit_status()
    sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    return status


def quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def first_login() -> int:
    """Answer the forced password change; print prompts, never answers."""
    env = read_credentials()
    temp, new = env["EEE_temp_password"], env["EEE_PASSWORD"]
    transport = open_transport(env["EEE_HOST"])

    seen = {"login": 0, "new": 0}

    def answer(prompt: str) -> str:
        p = prompt.lower()
        if "new" in p or "retype" in p:
            seen["new"] += 1
            if seen["new"] > 2:
                raise RuntimeError("New password refused; stopping.")
            return new
        if "current" not in p:
            seen["login"] += 1
            if seen["login"] > 1:
                raise RuntimeError("Temp password refused; stopping before lockout.")
        return temp  # "Password:" and "Current Password:"

    def handler(title, instructions, prompts):
        for prompt, _echo in prompts:
            print(f"[prompt] {title!r} {instructions!r} {prompt!r}")
        return [answer(prompt) for prompt, _echo in prompts]

    try:
        transport.auth_interactive(env["EEE_USERNAME"], handler)
    except paramiko.SSHException as exc:
        print(f"[auth] {type(exc).__name__}: {exc}")
    print(f"[auth] authenticated={transport.is_authenticated()}")
    if transport.is_authenticated():
        # Some PAM setups ask for the change inside the session instead.
        chan = transport.open_session(timeout=20)
        chan.get_pty()
        chan.invoke_shell()
        buffer = ""
        deadline = time.time() + 40
        while time.time() < deadline and not chan.closed:
            if chan.recv_ready():
                chunk = chan.recv(4096).decode("utf-8", errors="replace")
                buffer += chunk
                print(f"[shell] {chunk!r}")
                last = buffer.strip().splitlines()[-1].lower() if buffer.strip() else ""
                if last.endswith("password:"):
                    chan.send(answer(last) + "\n")
                    buffer = ""
                elif last.endswith(("$", "#", ">")):
                    chan.send("exit\n")
            else:
                time.sleep(0.3)
        chan.close()
    transport.close()
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["first-login"]:
        sys.exit(first_login())
    if sys.argv[1:2] == ["run"]:
        t = connect()
        try:
            sys.exit(run(t, " ".join(sys.argv[2:])))
        finally:
            t.close()
    print(__doc__)
    sys.exit(2)
