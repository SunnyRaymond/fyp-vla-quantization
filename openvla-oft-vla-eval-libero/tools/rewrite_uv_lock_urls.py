import csv
import sys
from pathlib import Path


lock_path = Path(sys.argv[1])
wheelhouse = Path(sys.argv[2]).resolve()
text = lock_path.read_text(encoding="utf-8")
rewritten = 0

with (wheelhouse / "selected-wheels.tsv").open(encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        old = row["url"]
        if old not in text:
            raise SystemExit(f"locked URL missing: {old}")
        text = text.replace(old, (wheelhouse / row["filename"]).as_uri(), 1)
        rewritten += 1

lock_path.write_text(text, encoding="utf-8")
print(f"rewrote {rewritten} locked wheel URLs")
