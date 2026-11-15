"""One-off edit of CVPR2027/figures/figure3_a.pptx, which is edited by hand and is the source of figure3_a.pdf:
removes the two rows of arrows above the bars (Fig. 2(b) shows the same picture), stretches the bar chart over the
height they leave free and keeps the word "frozen" with a swatch as the key of the bars. Run once."""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

PATH = Path(__file__).resolve().parents[3] / "CVPR2027/figures/figure3_a.pptx"
K = 1.72                                       # vertical stretch of the chart about its x axis
RX, START, END = 0.10, 285.0, 60.0             # the arcs, as in fig3_gap_arrows.py
KEY_Y, SWATCH = 0.16, 0.085                    # centre line of the key and side of its swatch, in inches
EMU = 914400

deck = Presentation(PATH)
shapes = deck.slides[0].shapes
named = {s.name: s for s in shapes}
assert "Connector 4" in named, "the arrows are already gone"

# the two rows of arrows and their labels fill the top 0.45 in; "frozen" stays as the key
key = named["TextBox 3"]
for shape in list(shapes):
    if shape.name != key.name and shape.top + shape.height < Inches(0.45):
        shape._element.getparent().remove(shape._element)

base = named["Connector 48"].top               # the x axis
stretch = lambda y: round(base - K * (base - y))  # noqa: E731


def recentre(shape):
    centre = shape.top + shape.height / 2
    shape.top = round(shape.top + stretch(centre) - centre)


for name in ("Connector 28", "Connector 30", "Connector 32", "Connector 34",   # ticks
             "TextBox 29", "TextBox 31", "TextBox 33", "TextBox 35",           # their labels
             "TextBox 27"):                                                    # "future error"
    recentre(named[name])
axis = named["Connector 47"]
axis.top = stretch(axis.top)
axis.height = base - axis.top
reference, label = named["Connector 36"], named["TextBox 37"]                  # the line at 1 and "redrawn"
shift = stretch(reference.top) - reference.top
reference.top, label.top = reference.top + shift, label.top + shift

bars = sorted((s for s in shapes if s.name.startswith("Round Same Side Corner Rectangle")), key=lambda s: s.left)
values = sorted((s for s in shapes if s.has_text_frame and "×" in s.text_frame.text), key=lambda s: s.left)
arcs = sorted((s for s in shapes if s.name.startswith("Gap arrow")), key=lambda s: s.left)
for bar, value, arc in zip(bars, values, arcs):
    height = round(K * bar.height)
    value.top = value.top - (height - bar.height)
    bar.top, bar.height = base - height, height
    top, bottom = bar.top / EMU + 0.015, reference.top / EMU - 0.015           # where the arc starts and ends
    ry = (bottom - top) / 1.832                # sin(60 deg) + sin(75 deg): the arc runs from 285 deg to 60 deg
    cx = (bar.left + bar.width) / EMU + 0.03 - 0.259 * RX
    arc.left, arc.top, arc.width, arc.height = Inches(cx - RX), Inches(top + 0.966 * ry - ry), Inches(2 * RX), Inches(2 * ry)

# the key of the bars, right-aligned with "redrawn"
key.left, key.top = label.left + label.width - key.width, Inches(KEY_Y) - key.height // 2
key.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT
swatch = shapes.add_shape(MSO_SHAPE.RECTANGLE, key.left - Inches(SWATCH - 0.005), Inches(KEY_Y - SWATCH / 2), Inches(SWATCH), Inches(SWATCH))
swatch.name = "Frozen swatch"
swatch.shadow.inherit = False
swatch.fill.solid()
swatch.fill.fore_color.rgb = RGBColor.from_string("B64342")
swatch.line.fill.background()
deck.save(PATH)
print("edited", PATH)
