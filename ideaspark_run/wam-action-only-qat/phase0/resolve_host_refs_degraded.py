"""Run the bundled resolver with the observed-failed SS engine excluded locally."""
import runpy
import sys
from pathlib import Path

phase0_dir = Path(__file__).resolve().parent
skill_script = Path('C:/Users/Raymond/.codex/skills/idea-spark/scripts/run.py')
sys.modules['scripts.search_semanticscholar'] = None
sys.argv = [str(skill_script), 'add_host_refs', '--out', str(phase0_dir),
            '--refs', str(phase0_dir / 'host_refs_nominations.json')]
print('DEGRADED host-ref lookup: skip Semantic Scholar after observed HTTP 429 in both windows; use bundled arXiv/OpenAlex title verification.', flush=True)
runpy.run_path(str(skill_script), run_name='__main__')
