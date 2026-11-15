"""Export panel (a) of the Fig. 1 slide as CVPR2027/figures/figure1_a.pdf, the teaser of the paper.

    python research/paper/figures/export_fig1a.py [deck.pptx] [out.pdf] [preview.png]

The slide also holds panels (b) and (c), which are not in the paper: a copy without them is exported through PowerPoint
and the page is cut to panel (a), the same box as the figure had so far.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz
from pptx import Presentation
from pptx.util import Inches

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parents[2] / "CVPR2027/figures"
BOX = (0.021, 0.021, 10.046, 7.542)  # panel (a) on the slide, in inches

source = Path(sys.argv[1]) if len(sys.argv) > 1 else FIGURES / "figure1_lavq_editable.pptx"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else FIGURES / "figure1_a.pdf"
work = Path(tempfile.mkdtemp())
deck = Presentation(source)
for shape in list(deck.slides[0].shapes):
    if shape.left >= Inches(BOX[2] - 0.02):
        shape._element.getparent().remove(shape._element)
deck.save(work / "panel_a.pptx")
subprocess.run([sys.executable, str(HERE / "export_pptx.py"), str(work / "panel_a.pptx"), str(work / "slide.pdf")], check=True)

slide = fitz.open(work / "slide.pdf")
clip = fitz.Rect(*(72 * v for v in BOX))
page_pdf = fitz.open()
page = page_pdf.new_page(width=clip.width, height=clip.height)
page.show_pdf_page(page.rect, slide, 0, clip=clip)
page_pdf.save(out, garbage=4, deflate=True)
if len(sys.argv) > 3:
    page.get_pixmap(dpi=300).save(sys.argv[3])
print("wrote", out)
