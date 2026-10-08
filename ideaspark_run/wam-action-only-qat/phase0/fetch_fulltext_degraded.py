"""Run bundled fulltext gate without repeatedly calling the failed SS endpoint."""
import runpy
import sys
from pathlib import Path

phase0_dir = Path(__file__).resolve().parent
skill_dir = Path('C:/Users/Raymond/.codex/skills/idea-spark')
sys.path.insert(0, str(skill_dir))
from scripts import fetch_sections

# Existing arXiv DOIs already contain their identifier; no new cross-reference is inferred.
fetch_sections._resolve_arxiv_via_ss = lambda doi, ss_id: (doi[len('10.48550/arxiv.'):] if doi and doi.lower().startswith('10.48550/arxiv.') else None)
fetch_sections._resolve_oa_pdf_via_ss = lambda doi: None
sys.argv = [str(skill_dir / 'scripts/run.py'), 'phase0_fulltext', '--out', str(phase0_dir)]
print('DEGRADED fulltext lookup: skip failed Semantic Scholar DOI/OA lookups; retain explicit arXiv IDs/DOIs and bundled HTML/PDF fetch/cache/gate.', flush=True)
runpy.run_path(sys.argv[0], run_name='__main__')
