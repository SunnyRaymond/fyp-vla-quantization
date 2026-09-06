"""Reject staged files larger than 50 MiB before archiving."""
import subprocess
import sys

entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"]).split(b"\0")
failed = False
blobs = []
for entry in entries:
    if not entry:
        continue
    metadata, path = entry.split(b"\t", 1)
    mode, oid, stage = metadata.split()
    if mode == b"160000":
        print("Source snapshot missing (gitlink):", path.decode())
        failed = True
        continue
    blobs.append((oid, path))
sizes = subprocess.check_output(
    ["git", "cat-file", "--batch-check=%(objectsize)"],
    input=b"\n".join(oid for oid, _ in blobs) + b"\n",
).splitlines() if blobs else []
for (_, path), raw_size in zip(blobs, sizes):
    size = int(raw_size)
    if size > 50 * 1024 * 1024:
        print(f"Too large ({size / 1024**2:.1f} MiB): {path.decode()}")
        failed = True
sys.exit(1 if failed else 0)
