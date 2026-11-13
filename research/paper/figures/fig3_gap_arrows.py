"""One-off edit of CVPR2027/figures/figure3_a.pptx, which is edited by hand and is the source of figure3_a.pdf:
narrows the three bars and adds, next to each, an arc with an arrowhead from the top of the bar down to the
"redrawn" reference line. Run once; the arcs are ordinary PowerPoint arc shapes and can be moved or reshaped."""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

PATH = Path(__file__).resolve().parents[3] / "CVPR2027/figures/figure3_a.pptx"
BAR, RX, START, END = 0.24, 0.10, 285.0, 60.0  # bar width and horizontal radius in inches; arc angles in degrees
EMU = 914400

deck = Presentation(PATH)
shapes = deck.slides[0].shapes
assert not any(s.name.startswith("Gap arrow") for s in shapes), "the arrows are already there"
bars = [s for s in shapes if s.name.startswith("Round Same Side Corner Rectangle")]
baseline = next(s for s in shapes if s.shape_type == 9 and s.width > Inches(1.5) and s.top < bars[0].top + bars[0].height - 1000)
for k, bar in enumerate(bars):
    centre = bar.left + bar.width / 2
    bar.left, bar.width = int(centre - Inches(BAR) / 2), Inches(BAR)
    top, bottom = bar.top / EMU + 0.015, baseline.top / EMU - 0.015  # where the arc starts and ends
    ry = (bottom - top) / 1.832               # sin(60 deg) + sin(75 deg): the arc runs from 285 deg to 60 deg
    cx = (bar.left + bar.width) / EMU + 0.03 - 0.259 * RX
    cy = top + 0.966 * ry
    arc = shapes.add_shape(MSO_SHAPE.ARC, Inches(cx - RX), Inches(cy - ry), Inches(2 * RX), Inches(2 * ry))
    arc.name = f"Gap arrow {k + 1}"
    arc.adjustments[0], arc.adjustments[1] = START * 0.6, END * 0.6
    arc.fill.background()
    arc.line.color.rgb, arc.line.width = RGBColor.from_string("222222"), Pt(0.8)
    outline = arc.line._get_or_add_ln()
    outline.append(outline.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "sm", "len": "sm"}))
    outline.addnext(outline.makeelement(qn("a:effectLst"), {}))
deck.save(PATH)
print("edited", PATH)
