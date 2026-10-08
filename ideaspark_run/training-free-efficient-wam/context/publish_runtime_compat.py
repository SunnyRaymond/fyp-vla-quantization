"""Run-local compatibility for the installed publication/renderer API mismatch.

Preserves the original validation, rendering, compilation, and receipt paths.
The caller requests PDF compilation; the installed renderer already does so.
"""

from pathlib import Path
import runpy
import sys

SKILL_ROOT = Path(r"C:\Users\Raymond\.codex\skills\idea-spark")
CLI = SKILL_ROOT / "scripts" / "run.py"
sys.path.insert(0, str(SKILL_ROOT))

from scripts import render_pdf

_original_render_one = render_pdf.render_one


def _render_one_compatible(expansion, out_dir, *, compile_pdfs=True):
    if compile_pdfs is not True:
        raise ValueError("This compatibility adapter preserves PDF compilation only")
    return _original_render_one(expansion, out_dir)


render_pdf.render_one = _render_one_compatible
sys.argv = [str(CLI), *sys.argv[1:]]
runpy.run_path(str(CLI), run_name="__main__")
