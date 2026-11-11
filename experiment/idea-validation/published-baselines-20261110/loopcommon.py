"""Stdlib helpers shared by the two closed-loop workers."""
from __future__ import annotations


def claim(root, order, start, worker):
    """Next unclaimed row at or after position `start`, by an atomic mkdir of claims/<index>."""
    claims = root / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    for k in range(start, len(order)):
        target = claims / str(order[k]["index"])
        try:
            target.mkdir()
        except FileExistsError:
            continue
        (target / f"worker_{worker}").touch()
        return order[k], k + 1
    return None, len(order)


def walltime_seconds(text):
    h, m, s = (int(x) for x in text.split(":"))
    return 3600 * h + 60 * m + s
