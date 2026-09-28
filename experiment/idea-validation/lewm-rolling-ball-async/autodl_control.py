"""Small SSH controls for the explicitly rented AutoDL host; never prints credentials."""
import argparse
import base64
import hashlib
import json
import shlex
import socket
import sys
from pathlib import Path

import paramiko

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "nscc-access"))
from aspire2a_shell import read_credentials

KNOWN_HOSTS = Path.home() / ".ssh" / "known_hosts_autodl_rolling"


def endpoint(command):
    command = command.strip()
    if command[:1] == command[-1:] and command[:1] in ("'", '"'):
        command = command[1:-1]
    tokens = shlex.split(command)
    if not tokens or tokens.pop(0) != "ssh":
        raise ValueError("SSH_COMMAND must be an ssh command")
    port, target = 22, None
    while tokens:
        token = tokens.pop(0)
        if token == "-p" and tokens:
            port = int(tokens.pop(0))
        elif token == "-o" and tokens:
            tokens.pop(0)  # Use this helper's host-key policy, never a supplied bypass.
        elif token.startswith("-") or target is not None:
            raise ValueError("Unsupported SSH_COMMAND option or remote command")
        else:
            target = token
    if not target or target.count("@") != 1 or not 1 <= port <= 65535:
        raise ValueError("Expected user@host and a valid port")
    user, host = target.split("@")
    if not user or not host or any(c.isspace() for c in host):
        raise ValueError("Invalid SSH endpoint")
    return host, port, user


def connect():
    credentials = read_credentials()
    host, port, user = endpoint(credentials["SSH_COMMAND"])
    password = credentials["AUTODL_PASSWORD"].strip().strip("\"'")
    transport = paramiko.Transport(socket.create_connection((host, port), timeout=25))
    try:
        transport.start_client(timeout=25)
        key = transport.get_remote_server_key()
        name = host if port == 22 else f"[{host}]:{port}"
        known = paramiko.HostKeys()
        for path in (Path.home() / ".ssh" / "known_hosts", KNOWN_HOSTS):
            if path.is_file():
                known.load(str(path))
        expected = known.lookup(name)
        if expected and not known.check(name, key):
            raise RuntimeError("Rented host-key mismatch; connection stopped")
        transport.auth_password(user, password)
        if not transport.is_authenticated():
            raise RuntimeError("AutoDL authentication failed")
        if not expected:
            # First-use enrollment is scoped to the user's explicit credentials endpoint.
            pinned = paramiko.HostKeys(str(KNOWN_HOSTS)) if KNOWN_HOSTS.is_file() else paramiko.HostKeys()
            pinned.add(name, key.get_name(), key)
            KNOWN_HOSTS.parent.mkdir(parents=True, exist_ok=True)
            pinned.save(str(KNOWN_HOSTS))
        metadata = {"host": host, "port": port, "user": user,
                    "known_host_line": f"{name} {key.get_name()} {key.get_base64()}",
                    "host_key_sha256": base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")}
        return transport, metadata
    except Exception:
        transport.close()
        raise


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--command")
    group.add_argument("--put", nargs=2)
    group.add_argument("--get", nargs=2)
    group.add_argument("--self-check", action="store_true")
    parser.add_argument("--save-endpoint", type=Path)
    parser.add_argument("--download-limit-mib", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.download_limit_mib <= 128:
        parser.error("download limit must be between 1 and 128 MiB")
    if args.self_check:
        assert endpoint("ssh -p 2222 root@example.invalid") == ("example.invalid", 2222, "root")
        assert endpoint('"ssh root@example.invalid"') == ("example.invalid", 22, "root")
        try:
            endpoint("ssh root@example.invalid touch bad")
        except ValueError:
            pass
        else:
            raise AssertionError("Remote command must be refused")
        print("PASS_endpoint_parser")
        return 0
    transport, metadata = connect()
    try:
        if args.save_endpoint:
            args.save_endpoint.parent.mkdir(parents=True, exist_ok=True)
            args.save_endpoint.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if args.put or args.get:
            with paramiko.SFTPClient.from_transport(transport) as sftp:
                sftp.get_channel().settimeout(60)
                if args.put:
                    local, remote = args.put
                    if Path(local).stat().st_size > 256 * 1024:
                        raise ValueError("Only small source/control uploads")
                    sftp.put(local, remote)
                else:
                    remote, local = args.get
                    if sftp.stat(remote).st_size > args.download_limit_mib * 1024 * 1024:
                        raise ValueError("Artifact exceeds the explicit download size limit")
                    target = Path(local)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_name(target.name + ".partial")
                    sftp.get(remote, str(partial))
                    partial.replace(target)
            return 0
        with transport.open_session(timeout=25) as channel:
            channel.settimeout(60)
            channel.exec_command(args.command)
            sys.stdout.write(channel.makefile("rb").read().decode("utf-8", errors="replace"))
            sys.stderr.write(channel.makefile_stderr("rb").read().decode("utf-8", errors="replace"))
            return channel.recv_exit_status()
    finally:
        transport.close()


if __name__ == "__main__":
    raise SystemExit(main())
