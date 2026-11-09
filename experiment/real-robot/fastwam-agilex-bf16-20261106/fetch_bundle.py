"""Download release/agilex_fastwam_bf16 from ASPIRE2A into deliver/agilex_fastwam_bf16 and verify it.

The file list and hashes come from the bundle's SHA256SUMS (written by manifest.pbs on a compute node). Partial files
resume from their current size; every file is hashed locally after download and must match. Symlinks in the remote
bundle (checkpoint, VAE) are read through, so the local copy holds the real files. User-approved login-node SFTP
transfer (no data-transfer node on ASPIRE2A).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import control as c  # noqa: E402

REMOTE = c.ROOT + "/release/agilex_fastwam_bf16"
LOCAL = HERE / "deliver" / "agilex_fastwam_bf16"
BLOCK = 8 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def main():
    jump, t = c.base.direct_connect()
    t0, report = time.time(), []
    try:
        with c.remote.sftp_client(t) as sftp:
            with sftp.open(REMOTE + "/SHA256SUMS", "rb") as f:
                sums = [line.split(None, 1) for line in f.read().decode().splitlines() if line.strip()]
            # manifest.pbs listed its own temporary output (SHA256SUMS.tmp, deleted afterwards); it is not bundle content
            sums = [(h, p.strip()) for h, p in sums if p.strip() not in ("SHA256SUMS", "SHA256SUMS.tmp")]
            (LOCAL).mkdir(parents=True, exist_ok=True)
            (LOCAL / "SHA256SUMS").write_text("".join(f"{h}  {p.strip()}\n" for h, p in sums), encoding="utf-8")
            for digest, rel in sums:
                rel = rel.strip()
                dst = LOCAL / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                size = sftp.stat(REMOTE + "/" + rel).st_size  # stat follows symlinks
                have = dst.stat().st_size if dst.exists() else 0
                if have > size:
                    dst.unlink()
                    have = 0
                if have < size:
                    s0 = time.time()
                    with sftp.open(REMOTE + "/" + rel, "rb") as src, dst.open("ab") as out:
                        src.seek(have)
                        src.prefetch(size)
                        done = have
                        while done < size:
                            block = src.read(min(BLOCK, size - done))
                            if not block:
                                raise IOError(f"short read on {rel} at {done}/{size}")
                            out.write(block)
                            done += len(block)
                    mb = (size - have) / 2**20
                    print(f"got {rel} {mb:.1f} MiB in {time.time() - s0:.0f}s ({mb / max(time.time() - s0, 1e-3):.1f} MiB/s)",
                          flush=True)
                ok = sha256(dst) == digest
                report.append(dict(file=rel, bytes=size, sha256_ok=ok))
                if not ok:
                    dst.unlink()
                    raise RuntimeError(f"sha256 mismatch on {rel}; removed, rerun to fetch it again")
    finally:
        t.close()
        if jump:
            jump.close()
    total = sum(r["bytes"] for r in report)
    print(json.dumps(dict(local=str(LOCAL), files=len(report), bytes=total, all_sha256_ok=all(r["sha256_ok"] for r in report),
                          seconds=round(time.time() - t0)), indent=2))


if __name__ == "__main__":
    main()
