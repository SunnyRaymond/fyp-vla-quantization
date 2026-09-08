import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path


expected = int(sys.argv[1])
roots = [Path(arg) for arg in sys.argv[2:]]
assert expected in {250, 500} and roots
rows = []
aggregate_total = 0
videos = []
for root in roots:
    dbs = list(root.glob("recording-*.sqlite"))
    assert len(dbs) == 1, dbs
    with sqlite3.connect(dbs[0]) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        rows += db.execute(
            "SELECT task_name, episode_id, status FROM episode_results"
        ).fetchall()
    aggregates = list(root.glob("*_aggregate.json"))
    assert len(aggregates) == 1, aggregates
    aggregate = json.loads(aggregates[0].read_text(encoding="utf-8"))
    aggregate_total += aggregate["num_episodes_total"]
    videos += list(root.rglob("*.mp4"))
counts = Counter(task for task, _, _ in rows)
pairs = {(task, episode_id) for task, episode_id, _ in rows}
assert len(rows) == len(pairs) == expected, (len(rows), len(pairs))
assert len(counts) == 10, counts
if expected == 500:
    assert set(counts.values()) == {50}, counts
    for task in counts:
        assert {episode_id for name, episode_id, _ in rows if name == task} == set(range(50))
else:
    assert set(counts.values()) == {24, 26}, counts
assert all(status in {"success", "fail", "error"} for _, _, status in rows)
assert aggregate_total == expected
assert len(videos) == expected and min(video.stat().st_size for video in videos) > 48
successes = sum(status == "success" for _, _, status in rows)
print(json.dumps({"episodes": len(rows), "tasks": len(counts), "videos": len(videos), "mean_success": successes / expected}))
