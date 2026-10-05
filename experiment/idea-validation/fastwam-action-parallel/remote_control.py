"""Small control-file transfer and PBS queries only; never run model work here."""
from __future__ import annotations

import argparse
import json
import shlex
import socket
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
PROJECT = LOCAL.parents[2]
sys.path.insert(0, str(PROJECT / "nscc-access"))
import aspire2a_shell as access
import paramiko

REMOTE = "/scratch/users/ntu/yguo017/fastwam-action-parallel"


def connect():
    try:
        return access.connect()
    except (OSError, EOFError, paramiko.SSHException) as exc:
        print(f"Jump route unavailable ({type(exc).__name__}); trying trusted direct route.")
        credentials = access.read_credentials()
        channel = paramiko.Transport(socket.create_connection((access.ASPIRE2A_HOST, 22), timeout=20))
        channel.start_client(timeout=20)
        access.verify_host_key(channel, access.ASPIRE2A_HOST, access.NSCC_KNOWN_HOSTS)
        channel.auth_password(credentials["NSCC_USERNAME"], credentials["NSCC_PASSWORD"])
        if not channel.is_authenticated():
            channel.close()
            raise RuntimeError("Direct authentication failed")
        return None, channel


def command(transport, value):
    session = transport.open_session(timeout=20)
    session.settimeout(45)
    session.exec_command(value)
    output = session.makefile("rb").read().decode("utf-8", errors="replace")
    error = session.makefile_stderr("rb").read().decode("utf-8", errors="replace")
    code = session.recv_exit_status()
    if error:
        print(error.rstrip(), file=sys.stderr)
    return code, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "upload", "submit", "pull"])
    parser.add_argument("--job")
    parser.add_argument("--line", choices=["context", "time"])
    args = parser.parse_args()
    jump, transport = connect()
    try:
        if args.action == "status":
            query = "qstat -fx " + shlex.quote(args.job) if args.job else "qstat -u yguo017"
            code, output = command(transport, query)
            print(output.rstrip())
            if args.job:
                _, tail = command(transport, "test ! -f " + shlex.quote(f"{REMOTE}/artifacts/{args.job}/job.log") + " || tail -c 5000 " + shlex.quote(f"{REMOTE}/artifacts/{args.job}/job.log"))
                print(tail.rstrip())
            return code
        if args.action == "upload":
            code, output = command(transport, "mkdir -p " + shlex.quote(REMOTE))
            if code:
                raise RuntimeError(output)
            with paramiko.SFTPClient.from_transport(transport) as sftp:
                for path in sorted(LOCAL.iterdir()):
                    if path.suffix not in {".py", ".pbs", ".md"} or path.name == "remote_control.py":
                        continue
                    if path.stat().st_size > 256 * 1024:
                        raise RuntimeError(f"Not a small control file: {path.name}")
                    data = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
                    with sftp.file(f"{REMOTE}/{path.name}", "wb") as target:
                        target.write(data)
                    print(f"Uploaded {path.name}: {len(data)} bytes")
            return 0
        if args.action == "submit":
            if not args.line:
                parser.error("submit requires --line")
            code, output = command(transport, f"qsub -v FW_LINE={args.line} " + shlex.quote(f"{REMOTE}/run.pbs"))
            print(output.rstrip())
            if code == 0:
                job = output.strip().splitlines()[-1]
                (LOCAL / f"{args.line}_job.json").write_text(json.dumps({"line": args.line, "job_id": job, "remote_root": REMOTE}, indent=2), encoding="utf-8")
            return code
        if not args.job:
            parser.error("pull requires --job")
        destination = LOCAL / "artifacts" / args.job
        destination.mkdir(parents=True, exist_ok=True)
        code, output = command(transport, "qstat -fx " + shlex.quote(args.job))
        (destination / "qstat.txt").write_text(output, encoding="utf-8")
        with paramiko.SFTPClient.from_transport(transport) as sftp:
            for name in ["summary.json", "measurements.csv", "resolved_config.yaml", "self_check.json", "exit_code.txt", "PIPELINE_COMPLETE", "job.log"]:
                remote = f"{REMOTE}/artifacts/{args.job}/{name}"
                try:
                    size = sftp.stat(remote).st_size
                except FileNotFoundError:
                    continue
                if size > 256 * 1024:
                    print(f"Skipped large artifact {name}: {size} bytes")
                    continue
                sftp.get(remote, str(destination / name))
                print(f"Retrieved {name}: {size} bytes")
        return code
    finally:
        transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
