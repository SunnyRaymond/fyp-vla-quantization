"""Wait for aspire_pack.pbs, then stream its tar and manifests from ASPIRE2A to EEE through this machine.

Resumes a partial file after a dropped connection. Run from the project root:
  nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/laq-libero840-20261112/relay.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "nscc-access"), str(ROOT / "eee-access")]
import aspire2a_shell as aspire  # noqa: E402
import eee_shell as eee  # noqa: E402

SRC = "/scratch/users/ntu/yguo017/eee-migration-laq840"
DST = "/projects/yguo017/incoming/laq840"
NAMES = ["SHA256SUMS", "FILES.sha256", "members.txt", "laq840.tar"]
CHUNK = 1 << 20


def size_of(sftp, path):
    try:
        return sftp.stat(path).st_size
    except FileNotFoundError:
        return 0


def wait_for_pack():
    deadline = time.time() + 3 * 3600
    while time.time() < deadline:
        jump, nscc = aspire.connect()
        try:
            if size_of(paramiko.SFTPClient.from_transport(nscc), f"{SRC}/SHA256SUMS"):
                print("PACK_READY", flush=True)
                return
        finally:
            nscc.close()
            jump.close()
        time.sleep(60)
    raise TimeoutError("pack job did not finish")


def relay_one(name):
    jump, nscc = aspire.connect()
    target = eee.connect()
    try:
        src = paramiko.SFTPClient.from_transport(nscc)
        dst = paramiko.SFTPClient.from_transport(target)
        try:
            dst.stat(DST)
        except FileNotFoundError:
            dst.mkdir(DST)
        total = src.stat(f"{SRC}/{name}").st_size
        if size_of(dst, f"{DST}/{name}") == total:
            print("HAVE", name, flush=True)
            return
        part = f"{DST}/{name}.part"
        done = size_of(dst, part)
        started, start_done, mark = time.time(), done, done
        with src.open(f"{SRC}/{name}", "rb") as reader, dst.open(part, "ab") as writer:
            reader.seek(done)
            reader.prefetch(total - done)
            writer.set_pipelined(True)
            while done < total:
                data = reader.read(CHUNK)
                if not data:
                    raise IOError(f"short read on {name} at {done}")
                writer.write(data)
                done += len(data)
                if done - mark >= 1 << 29:
                    mark = done
                    rate = (done - start_done) / max(time.time() - started, 1) / 1e6
                    print(f"PROGRESS {name} {done / 1e9:.2f}/{total / 1e9:.2f} GB {rate:.1f} MB/s", flush=True)
        dst.posix_rename(part, f"{DST}/{name}")
        print("DONE", name, total, flush=True)
    finally:
        nscc.close()
        jump.close()
        target.close()


def retry(name):
    for attempt in range(10):
        try:
            relay_one(name)
            return
        except (OSError, EOFError, paramiko.SSHException) as exc:
            print(f"RETRY {attempt} {name} {type(exc).__name__}: {exc}", flush=True)
            time.sleep(30)
    raise RuntimeError(f"gave up on {name}")


if __name__ == "__main__":
    wait_for_pack()
    for item in NAMES:
        retry(item)
    print("RELAY_COMPLETE", flush=True)
