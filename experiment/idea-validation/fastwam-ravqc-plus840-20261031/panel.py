"""Frozen LIBERO-Plus panels A, B, C (280 variants each, slot indices 0-279 / 280-559 / 560-839), read-only: the
manifests written by fastwam-rotation-baselines (A), fastwam-plus-panelb-20261031 (B) and fastwam-plus-panelc-20261031
(C). Stdlib only (host python3.11 and the container both import it)."""
from __future__ import annotations

import json
from pathlib import Path

PANEL_A_PROTOCOL = "fastwam-rotation-baselines-v1"
PANEL = 280
PANELS = ("A", "B", "C")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(p, name):
    """Frozen rows of panel A, B or C, checked against their protocol and slot indices."""
    if name == "A":
        path, protocol, offset = p["panel_a"]["manifest"], PANEL_A_PROTOCOL, 0
    elif name in ("B", "C"):
        c = p["panel_b" if name == "B" else "panel_c"]
        path, protocol, offset = c["manifest"], c["protocol"], c["index_offset"]
    else:
        raise ValueError(name)
    doc = read(path)
    rows = doc["variants"]
    if doc.get("protocol") != protocol or sorted(int(r["index"]) for r in rows) != list(range(offset, offset + PANEL)):
        raise RuntimeError(f"Panel {name} manifest {path} does not match protocol {protocol} / offset {offset}")
    return rows


def panel_of(index):
    return PANELS[int(index) // PANEL]
