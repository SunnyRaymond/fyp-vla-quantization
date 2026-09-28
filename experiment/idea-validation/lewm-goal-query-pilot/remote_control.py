"""Small control-file transfers and scheduler queries with pinned SSH keys."""
import argparse
import socket
import sys
from pathlib import Path

import paramiko

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "nscc-access"))
from aspire2a_shell import ASPIRE2A_HOST, NSCC_KNOWN_HOSTS, read_credentials, verify_host_key


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--command")
    group.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"))
    group.add_argument("--get", nargs=2, metavar=("REMOTE", "LOCAL"))
    args = parser.parse_args()
    credentials = read_credentials()
    transport = paramiko.Transport(socket.create_connection((ASPIRE2A_HOST, 22), timeout=20))
    try:
        transport.start_client(timeout=20)
        verify_host_key(transport, ASPIRE2A_HOST, NSCC_KNOWN_HOSTS)
        transport.auth_password(credentials["NSCC_USERNAME"], credentials["NSCC_PASSWORD"])
        if not transport.is_authenticated():
            raise RuntimeError("ASPIRE2A authentication failed")
        if args.put or args.get:
            with paramiko.SFTPClient.from_transport(transport) as sftp:
                sftp.get_channel().settimeout(60)
                if args.put:
                    local, remote = args.put
                    if Path(local).stat().st_size > 256 * 1024:
                        raise ValueError("Only small source/control uploads are permitted")
                    sftp.put(local, remote)
                else:
                    remote, local = args.get
                    if sftp.stat(remote).st_size > 2 * 1024 * 1024:
                        raise ValueError("Only small result reports may be retrieved")
                    target = Path(local)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_name(target.name + ".partial")
                    sftp.get(remote, str(partial))
                    partial.replace(target)
            return 0
        with transport.open_session(timeout=20) as channel:
            channel.settimeout(60)
            channel.exec_command(args.command)
            stdout = channel.makefile("rb").read().decode("utf-8", errors="replace")
            stderr = channel.makefile_stderr("rb").read().decode("utf-8", errors="replace")
            sys.stdout.write(stdout)
            sys.stderr.write(stderr)
            return channel.recv_exit_status()
    finally:
        transport.close()


if __name__ == "__main__":
    raise SystemExit(main())
