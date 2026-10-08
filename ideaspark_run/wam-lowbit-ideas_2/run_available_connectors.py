"""Run the installed IdeaSpark CLI with this run's bounded retrieval sources.

S2 timed out, then its coverage lookup returned repeated HTTP 429. OpenReview
is outside this trimmed run. Records still come from the installed arXiv and
OpenAlex connectors, and the CLI's title-match and grounding gates stay intact.
This wrapper changes only this process; installed skills and credentials stay
read-only. It makes no claim of comprehensive retrieval.
"""
import sys
import subprocess
import json
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(r"C:\Users\Raymond\.codex\skills\idea-spark")))
from scripts import search_semanticscholar, run


def omit_s2(*args, **kwargs):
    raise RuntimeError("Semantic Scholar omitted in this run after timeout/429; use arXiv/OpenAlex")


search_semanticscholar.search = omit_s2
run.CONNECTORS = [item for item in run.CONNECTORS if item[0] in {"arxiv", "openalex"}]

# quality_collision invokes phase3_collision in a child process. Keep the same
# explicitly limited sources there while retaining the installed receipt rules
# (non-empty real hits, unchanged input, official deterministic receipt).
_original_subprocess_run = subprocess.run


def bounded_collision_child(argv, *args, **kwargs):
    if (isinstance(argv, (list, tuple)) and len(argv) > 2
            and argv[2] == "phase3_collision"
            and Path(argv[1]).name == "run.py"):
        argv = [argv[0], str(Path(__file__).resolve()), *argv[2:]]
    return _original_subprocess_run(argv, *args, **kwargs)


subprocess.run = bounded_collision_child


def collision_cache_info(argv):
    if not argv or argv[0] != "phase3_collision":
        return None
    source = Path(argv[argv.index("--idea-json") + 1])
    output = Path(argv[argv.index("--out") + 1])
    payload = json.loads(source.read_text(encoding="utf-8"))
    key = {"signature_terms": payload.get("signature_terms", []),
           "alias_terms": payload.get("alias_terms", []),
           "day": datetime.now(timezone.utc).date().isoformat(),
           "sources": ["arxiv", "openalex"]}
    return output, key


def cli():
    info = collision_cache_info(sys.argv[1:])
    if info:
        output, key = info
        marker = output / "completed_retrieval_terms.json"
        hits = output / "collision_hits.json"
        if marker.exists() and hits.exists():
            if (json.loads(marker.read_text(encoding="utf-8")) == key
                    and json.loads(hits.read_text(encoding="utf-8"))):
                print("Reusing completed same-day arXiv/OpenAlex collision retrieval for identical terms.")
                return 0
    code = run.main()
    if info and code == 0:
        output, key = info
        hits = output / "collision_hits.json"
        if hits.exists() and json.loads(hits.read_text(encoding="utf-8")):
            (output / "completed_retrieval_terms.json").write_text(
                json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    return code

if __name__ == "__main__":
    sys.exit(cli())
