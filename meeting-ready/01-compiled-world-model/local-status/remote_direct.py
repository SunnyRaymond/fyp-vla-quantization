"""Small direct ASPIRE2A control helper using the existing pinned host key."""

import argparse
import stat
import sys
from pathlib import Path

import paramiko

ACCESS = Path(__file__).resolve().parents[3] / "nscc-access"
sys.path.insert(0, str(ACCESS))
from aspire2a_shell import ASPIRE2A_HOST, NSCC_KNOWN_HOSTS, read_credentials, verify_host_key


def connect():
    credentials = read_credentials()
    transport = paramiko.Transport((ASPIRE2A_HOST, 22))
    transport.start_client(timeout=20)
    verify_host_key(transport, ASPIRE2A_HOST, NSCC_KNOWN_HOSTS)
    transport.auth_password(credentials["NSCC_USERNAME"], credentials["NSCC_PASSWORD"])
    if not transport.is_authenticated():
        raise RuntimeError("ASPIRE2A direct authentication failed")
    return transport


parser = argparse.ArgumentParser()
parser.add_argument("--command")
parser.add_argument("--put", nargs=2)
parser.add_argument("--get", nargs=2)
parser.add_argument("--get-dir", nargs=2)
args = parser.parse_args()
transport = connect()
try:
    if args.put or args.get or args.get_dir:
        sftp = paramiko.SFTPClient.from_transport(transport)
        if args.put:
            sftp.put(*args.put)
        elif args.get:
            sftp.get(*args.get)
        else:
            source, destination = args.get_dir
            target = Path(destination)
            target.mkdir(parents=True, exist_ok=True)
            for entry in sftp.listdir_attr(source):
                if stat.S_ISREG(entry.st_mode):
                    sftp.get(source.rstrip("/") + "/" + entry.filename, str(target / entry.filename))
        sftp.close()
    else:
        channel = transport.open_session(timeout=20)
        channel.settimeout(300)
        channel.exec_command(args.command)
        stdout = channel.makefile("rb").read().decode("utf-8", errors="replace")
        stderr = channel.makefile_stderr("rb").read().decode("utf-8", errors="replace")
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
        raise SystemExit(channel.recv_exit_status())
finally:
    transport.close()
