"""Fig. 2, the overview of LAQ, as an editable PowerPoint slide.

The layout follows the drafts CVPR2027/figures/figure2_laq_gpt.png and figure2_laq_gpt2.png. Every coordinate
below is a pixel of those drafts (1774 x 887); one pixel is 0.01 in on the slide. Text, lines, grids and
arrows are native shapes.

    python research/paper/figures/fig2_method.py            # writes CVPR2027/figures/figure2_laq_editable.pptx
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE, MSO_PATTERN
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "CVPR2027" / "figures" / "figure2_laq_editable.pptx"
ASSETS = Path(__file__).resolve().parent / "figure3_assets"  # the frame and the map of Fig. 3(b)
W, H = 1774, 887
PX = 9144  # EMU per draft pixel

FONT, MATH = "Cambria", "Cambria Math"
INK, MUTED, RULE = "1A1A1A", "6B7280", "9A9AA0"
LINK = "1F78D1"                      # offline quantities handed to other blocks
BASE_EDGE, BASE = "4F8FDB", "CFE0F5"  # first 4-bit pass
RES_EDGE, RES = "E07B1A", "F9C66B"    # second 4-bit pass
GAZE = "C8641B"                      # run-time gaze feedback
FROZEN = "C0282D"
BUS = "D9E1EE"
HEAT = ["FFF6D6", "FDCB6E", "F28C38", "D8432B", "8C1020"]
BLUES = ["F2F6FC", "D5E3F6", "A9C6EC", "2F6FC4", "1F4E8C"]

deck = Presentation()
deck.slide_width, deck.slide_height = Emu(W * PX), Emu(H * PX)
shapes = deck.slides.add_slide(deck.slide_layouts[6]).shapes
E = lambda v: Emu(int(round(v * PX)))  # noqa: E731
rgb = RGBColor.from_string


def ramp(colors, t):
    t = min(max(t, 0.0), 1.0) * (len(colors) - 1)
    k = min(int(t), len(colors) - 2)
    a, b = (tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)) for c in (colors[k], colors[k + 1]))
    return "".join(f"{round(a[i] + (b[i] - a[i]) * (t - k)):02X}" for i in range(3))


def runs(markup):
    """`$` toggles italic, `*` bold, `@` the math font; `_{}` and `^{}` set a subscript or superscript."""
    out, italic, bold, math, buf, i = [], False, False, False, "", 0

    def flush(base=0, slant=None):
        nonlocal buf
        if buf:
            out.append((buf, italic if slant is None else slant, bold, math, base))
        buf = ""

    while i < len(markup):
        c = markup[i]
        if c in "$*@":
            flush()
            italic, bold, math = italic ^ (c == "$"), bold ^ (c == "*"), math ^ (c == "@")
        elif c in "_^" and markup[i + 1:i + 2] == "{":
            flush()
            end = markup.index("}", i)
            inner = markup[i + 2:end]  # a `$` inside the braces sets the whole index in italic
            buf = inner.replace("$", "")
            flush(-25000 if c == "_" else 30000, italic or "$" in inner)
            i = end
        else:
            buf += c
        i += 1
    flush()
    return out


def text(x, y, markup, size=13, color=INK, ha="center", bold=False, italic=False):
    """Text centred vertically on y; `ha` places it about x. Lines are separated by a newline."""
    rows = markup.split("\n")
    plain = max(len("".join(r[0] for r in runs(row))) for row in rows)
    w, h = plain * size * 1.389 * 0.62 + 14, len(rows) * size * 1.389 * 1.22 + 4
    left = {"center": x - w / 2, "left": x, "right": x - w}[ha]
    shape = shapes.add_textbox(E(left), E(y - h / 2), E(w), E(h))
    frame = shape.text_frame
    frame.word_wrap, frame.auto_size, frame.vertical_anchor = False, MSO_AUTO_SIZE.NONE, MSO_ANCHOR.MIDDLE
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    for k, row in enumerate(rows):
        paragraph = frame.paragraphs[0] if k == 0 else frame.add_paragraph()
        paragraph.alignment = {"center": PP_ALIGN.CENTER, "left": PP_ALIGN.LEFT, "right": PP_ALIGN.RIGHT}[ha]
        paragraph.line_spacing = 0.95
        for s, it, bd, math, base in runs(row):
            run = paragraph.add_run()
            run.text = s
            font = run.font
            font.name, font.size = (MATH if math else FONT), Pt(size)
            font.italic, font.bold = it ^ italic, bd ^ bold
            font.color.rgb = rgb(color)
            if base:
                run._r.get_or_add_rPr().set("baseline", str(base))
    return shape


def style(shape, fill=None, edge=None, lw=1.0, alpha=None, dash=False):
    shape.shadow.inherit = False
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill)
        if alpha is not None:
            node = shape.fill._xPr.find(qn("a:solidFill")).find(qn("a:srgbClr"))
            node.append(node.makeelement(qn("a:alpha"), {"val": str(int(alpha * 1000))}))
    else:
        shape.fill.background()
    if edge:
        shape.line.color.rgb, shape.line.width = rgb(edge), Pt(lw)
        if dash:
            shape.line.dash_style = MSO_LINE.DASH
    else:
        shape.line.fill.background()
    return shape


def shadow(shape, blur=5.0, dist=2.2, alpha=28):
    """A soft shadow below the shape, in points."""
    effects = shape._element.spPr.find(qn("a:effectLst"))
    outer = effects.makeelement(qn("a:outerShdw"), {"blurRad": str(int(blur * 12700)), "dist": str(int(dist * 12700)),
                                                    "dir": "5400000", "algn": "t", "rotWithShape": "0"})
    colour = outer.makeelement(qn("a:srgbClr"), {"val": "000000"})
    colour.append(colour.makeelement(qn("a:alpha"), {"val": str(alpha * 1000)}))
    outer.append(colour)
    effects.append(outer)
    return shape


def picture(x, y, w, h, name):
    return shapes.add_picture(str(ASSETS / name), E(x), E(y), E(w), E(h))


def rect(x, y, w, h, fill=None, edge=None, lw=1.0, radius=0.0, into=None, alpha=None):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = (into or shapes).add_shape(kind, E(x), E(y), E(w), E(h))
    if radius:
        shape.adjustments[0] = min(radius / min(w, h), 0.5)
    return style(shape, fill, edge, lw, alpha)


def heads(shape, end=True, start=False, size="med"):
    outline = shape.line._get_or_add_ln()
    if start:
        outline.append(outline.makeelement(qn("a:headEnd"), {"type": "triangle", "w": size, "len": size}))
    if end:
        outline.append(outline.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": size, "len": size}))


def line(x1, y1, x2, y2, color=INK, lw=1.4, arrow=False, both=False, dash=False, size="med"):
    shape = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, E(x1), E(y1), E(x2), E(y2))
    shape.line.color.rgb, shape.line.width = rgb(color), Pt(lw)
    if dash:
        shape.line.dash_style = MSO_LINE.DASH
    heads(shape, end=arrow or both, start=both, size=size)
    outline = shape.line._get_or_add_ln()
    outline.addnext(outline.makeelement(qn("a:effectLst"), {}))
    return shape


def path(points, color=INK, lw=1.4, arrow=True, dash=False, fill=None, alpha=None, close=False, size="med"):
    """An open polyline with an arrowhead at its last point, or a closed filled outline."""
    builder = shapes.build_freeform(E(points[0][0]), E(points[0][1]), scale=1.0)
    builder.add_line_segments([(E(x), E(y)) for x, y in points[1:]], close=close)
    shape = style(builder.convert_to_shape(), fill, color, lw, alpha, dash)
    if arrow and not close:
        heads(shape, size=size)
    return shape


def curve(p0, c, p1, n=16):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t * t * p1[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t * t * p1[1]) for t in (k / n for k in range(n + 1))]


def grid(x, y, w, h, colors, edge=RULE, lw=0.5, frame=INK, frame_lw=1.0):
    """A grid of filled cells (rows of hex colours) inside one group, with a frame around it."""
    group = shapes.add_group_shape()
    rows, cols = len(colors), len(colors[0])
    for r in range(rows):
        for c in range(cols):
            rect(x + w * c / cols, y + h * r / rows, w / cols, h / rows, colors[r][c], edge, lw, into=group.shapes)
    if frame:
        rect(x, y, w, h, None, frame, frame_lw, into=group.shapes)
    return group


def dot(x, y, r=4.5, color=INK):
    return style(shapes.add_shape(MSO_SHAPE.OVAL, E(x - r), E(y - r), E(2 * r), E(2 * r)), color)


def chip(x, y, w, h, label, fill, edge, size=13):
    shadow(rect(x, y, w, h, fill, edge, 1.2, radius=5), 3.0, 1.4, 22)
    text(x + w / 2, y + h / 2, label, size)


def brace(x, y, w, h):
    shape = shapes.add_shape(MSO_SHAPE.RIGHT_BRACE, E(x), E(y), E(w), E(h))
    return style(shape, None, INK, 1.3)


# ---- frame of the figure ---------------------------------------------------------------------------------
text(25, 17, "*Offline, once*", 17, ha="left")
line(22, 36, 1754, 36, lw=1.2)
rect(1282, 9, 16, 16, BASE, BASE_EDGE, 1.0)
text(1305, 17, "1st 4-bit pass", 12, ha="left")
rect(1428, 9, 16, 16, RES, RES_EDGE, 1.0)
text(1451, 17, "2nd 4-bit pass", 12, ha="left")
line(1578, 17, 1618, 17, GAZE, 2.0, dash=True)
text(1626, 17, "Gaze feedback", 12, ha="left")

# ---- (a) reader metric -----------------------------------------------------------------------------------
text(27, 62, "*(a) Reader metric*", 17, ha="left")
text(45, 131, "$z$^{∗}", 16)
for k in range(6):
    xl = 62 + 12.5 * k
    shadow(path([(xl, 118), (xl + 38, 93), (xl + 38, 145), (xl, 170)], "3B7DD8", 1.0, close=True, fill="DCE9FA"),
           3.0, 1.2, 18)
text(180, 130, "···", 13)
line(192, 131, 217, 131, arrow=True, size="sm")
shadow(rect(221, 112, 103, 39, "E4E4E8", RULE, 1.2, radius=9))
text(272, 131, "*Reader* $R$", 13)
line(330, 131, 360, 131, arrow=True, size="sm")
for k in range(3):
    shadow(rect(364 + 25 * k, 119, 21, 26, "C9C3E6", "6B6799", 1.0, radius=2), 3.0, 1.2, 20)
text(462, 131, "$a$^{∗}", 16)
path(curve((276, 152), (268, 186), (118, 176)), "1F5FC8", 1.6)
text(207, 196, "Reader probes", 13, "1F5FC8", italic=True)

text(24, 300, "$M$ = E_{$c$}[$J$_{$c$}^{$T$} $D$_{$c$} $J$_{$c$}]", 14, ha="left")
n = 6
grid(171, 237, 123, 129, [[BLUES[3] if r == c else BLUES[(r * 3 + c * 5) % 3] for c in range(n)] for r in range(n)],
     edge="FFFFFF")
line(231, 372, 231, 409, arrow=True, size="sm")
text(334, 268, "Token\nblock\ntrace", 13)
path(curve((296, 330), (330, 332), (336, 320), 8) + curve((336, 320), (342, 310), (367, 311), 8)[1:], INK, 1.3,
     size="sm")
text(436, 221, "*Token sensitivity*", 13)
picture(372, 238, 128, 128, "b_frame_1.png")     # third-person view of a predicted LIBERO frame, in grey
picture(372, 238, 128, 128, "b_overlay_1.png")   # share of tr(M) per token, as in Fig. 3(b)
rect(372, 238, 128, 128, None, INK, 1.0)
path([(261, 368), (261, 385), (549, 385), (549, 330)], LINK, 1.4)
text(549, 276, "Future\nerror", 12.5)
text(547, 311, "$Z$_{$M$}", 16)
line(572, 298, 590, 298)
dot(590, 298)
line(590, 298, 705, 298, arrow=True)
path([(590, 298), (590, 94), (1477, 94), (1477, 110)], LINK, 1.4)
text(968, 76, "$Z$_{$M$}", 15)

# ---- (b) read count --------------------------------------------------------------------------------------
text(637, 62, "*(b) Read count*", 17, ha="left")
text(637, 118, "Denoising steps", 13, ha="left")
STEP = [800, 850, 935]
for x, label in zip(STEP + [893, 998], ["1", "2", "$T$", "···", "*Sum*"]):
    text(x, 117, label, 13)
line(775, 139, 960, 139, lw=1.1)
for x in (775, 960):
    line(x, 139, x, 147, lw=1.1)
for x in STEP:
    line(x, 131, x, 147, lw=1.1)
line(968, 150, 968, 262, RULE, 0.8)
text(768, 153, "Equal-energy errors", 12, MUTED, ha="right", italic=True)
text(647, 180, "*Frozen*", 15, FROZEN, ha="left")
for x in STEP:
    line(x - 15, 180, x + 15, 180, FROZEN, 2.0, arrow=True)
text(893, 178, "···", 13)
line(976, 180, 1020, 180, FROZEN, 3.0, arrow=True)
text(843, 202, "Weights, $fixed tokens, reused K/V$", 12)
text(647, 233, "*Fresh*", 15, MUTED, ha="left")
for x, (dx, dy) in zip(STEP, [(14, 0), (10, -10), (-10, -10)]):
    line(x - dx, 233 - dy, x + dx, 233 + dy, MUTED, 1.6, arrow=True)
text(893, 231, "···", 13)
line(976, 233, 996, 233, MUTED, 2.0, arrow=True)
text(850, 256, "Updated activations", 12, italic=True)
grid(1030, 158, 48, 48, [["A8262B" if r == c else "CC5357" for c in range(4)] for r in range(4)], edge="FFFFFF")
grid(1030, 214, 48, 48, [[BLUES[4] if r == c else BLUES[0] for c in range(4)] for r in range(4)], edge="D5E3F6")
text(1092, 182, "Cross terms\ncan reinforce", 12.5, ha="left")
text(1092, 238, "Cross terms\naverage out", 12.5, ha="left")

text(722, 292, "$n$_{$u$} =", 16, ha="left")
text(838, 277, "$Z$_{$M$}(frozen)", 15)
line(780, 293, 897, 293, lw=1.2)
text(838, 309, "$Z$_{$M$}(fresh)", 15)
text(812, 337, "Fresh:  $n$_{$u$} = 1", 13.5)
line(912, 297, 1213, 297, arrow=True)
text(1110, 282, "$n$_{$u$}", 15)
dot(1002, 297)
line(1002, 297, 1002, 411, arrow=True, size="sm")
text(660, 372, "*Step profile*  $φ$_{$s$}", 13.5, ha="left")
line(825, 387, 985, 387, lw=1.1)
for x, w, top in [(858, 35, 375), (893, 31, 368), (924, 30, 362), (954, 29, 355)]:
    rect(x, top, w, 387 - top, "BBD3F2", "2A2A2A", 0.9)
line(920, 387, 920, 411, arrow=True, size="sm")
text(1018, 373, "Refine one step\nat a time", 12.5, ha="left")

# ---- (c) joint allocation --------------------------------------------------------------------------------
text(1215, 62, "*(c) Joint allocation*", 17, ha="left")
text(1495, 131, "@ℒ@ = E_{$c$} [$A$ + $λ$_{$M$} $Z$_{$M$} + $βK$]", 20)
line(1472, 158, 1472, 168, lw=1.4)
path([(1472, 168), (1445, 182), (1332, 182), (1332, 205)], INK, 1.4)
path([(1472, 168), (1500, 182), (1622, 182), (1622, 205)], INK, 1.4)
text(1341, 221, "*Activations and K/V*", 15)
text(1625, 221, "*Weights*", 15)
line(1220, 250, 1220, 340, lw=1.4)
line(1220, 340, 1442, 340, lw=1.4)
for k, top in enumerate([250, 265, 279, 294, 309, 320]):
    bar = rect(1236 + 33.5 * k, top, 26, 340 - top, ["7E828C", "8E929C", "A2A6AF"][min(k, 2)], "6B7280", 0.8)
    if k % 2:  # cached keys and values
        bar.fill.patterned()
        bar.fill.pattern = MSO_PATTERN.WIDE_UPWARD_DIAGONAL
        bar.fill.fore_color.rgb, bar.fill.back_color.rgb = rgb("7E828C"), rgb("FFFFFF")
line(1222, 300, 1430, 300, lw=1.2, dash=True)
text(1445, 293, "$μ$", 17)
rect(1352, 246, 13, 13, "8E929C", "6B7280", 0.8)
text(1371, 253, "Activation", 12, ha="left")
key = rect(1352, 265, 13, 13, None, "6B7280", 0.8)
key.fill.patterned()
key.fill.pattern = MSO_PATTERN.WIDE_UPWARD_DIAGONAL
key.fill.fore_color.rgb, key.fill.back_color.rgb = rgb("7E828C"), rgb("FFFFFF")
text(1371, 272, "K/V", 12, ha="left")
text(1338, 363, "Budget $B$ → shared threshold $μ$", 12.5)
line(1370, 380, 1370, 411, arrow=True, size="sm")
weights = [[BLUES[3] if r == c else BLUES[(r * 2 + c * 3) % 3] for c in range(4)] for r in range(4)]
grid(1499, 257, 64, 72, weights, edge="FFFFFF", frame="1F4E8C")
grid(1692, 257, 64, 72, weights, edge="FFFFFF", frame="1F4E8C")
text(1628, 273, "Second-order\nrounding", 12, italic=True)
line(1580, 300, 1682, 300, arrow=True)
text(1630, 356, "4-bit weights  $Q$_{4}($W$)", 14)
line(1638, 378, 1638, 411, arrow=True, size="sm")

# ---- what the offline stage hands over -------------------------------------------------------------------
shadow(rect(20, 417, 1734, 36, BUS, None, radius=6), 4.0, 1.6, 20)
text(252, 435, "Init.  tr($M$_{$jj$})", 14)
text(898, 435, "*Scoring parameters:*  $n$_{$u$}, $φ$_{$s$}, $μ$", 14)
text(1640, 435, "$Q$_{4}($W$)", 14)
path([(232, 453), (232, 476), (14, 476), (14, 612), (36, 612)], LINK, 1.4)
path([(882, 453), (882, 511), (445, 511), (445, 568)], LINK, 1.4)
path([(1640, 453), (1640, 476), (1487, 476), (1487, 576), (1335, 576), (1335, 596)], LINK, 1.4)

# ---- (d) gaze-guided refinement --------------------------------------------------------------------------
line(22, 489, 1762, 489, lw=1.2)
line(22, 535, 1762, 535, lw=0.7)
text(22, 511, "*(d) Gaze-guided refinement*", 17, ha="left")
text(1752, 511, "At every control step", 14, ha="right", italic=True)
rect(1040, 498, 290, 26, "F1F3F7", "C9CED8", 0.8, radius=13)
text(1185, 511, "*All weights 4-bit, no 16-bit branch*", 12.5)
for x in (552, 1192):
    line(x, 556, x, 766, RULE, 0.9)

# score and select
text(37, 554, "*Score and select*", 15, ha="left")
gaze = [0.05, 0.10, 0.25, 0.38, 0.62, 0.50]
profile = [0.35, 0.75, 1.0, 0.5]
text(40, 612, "Current gaze g_{j}", 13.5, ha="left", italic=True)
grid(39, 642, 128, 25, [[ramp(HEAT, g) for g in gaze]], edge="B9B9BE")
text(183, 653, "···", 13)
text(217, 653, "×", 17)
text(255, 586, "$φ$_{$s$}", 15)
grid(244, 604, 23, 87, [[ramp(BLUES[:3], p)] for p in profile], edge="6B7280")
text(296, 653, "=", 17)
text(327, 646, "$s$", 14)
text(417, 588, "$ω$_{$sj$} = $φ$_{$s$} $g$_{$j$}", 14)
grid(342, 608, 156, 74, [[ramp(HEAT, min(1.0, 1.45 * p * g)) for g in gaze] for p in profile], edge="E8D9B0")
text(420, 697, "$j$", 14)
text(70, 750, "$score$_{$u$} =", 15, ha="left")
text(198, 735, "$n$_{$u$} $ω$_{$u$} $σ$_{$u$}^{2}", 14)
line(165, 751, 232, 751, lw=1.2)
text(198, 766, "$c$_{$u$}", 14)
text(262, 750, "≥ $μ$", 16)
line(300, 757, 338, 757, arrow=True, size="sm")
text(430, 730, "Selected tokens", 12.5)
for k in range(6):
    chosen = k in (3, 4)
    rect(357 + 24 * k, 747, 24, 24, RES if chosen else BASE, RES_EDGE if chosen else BASE_EDGE, 1.6 if chosen else 0.9)
text(516, 762, "···", 13)
path([(512, 772), (512, 789), (1005, 789), (1005, 778)], RES_EDGE, 1.4)
text(830, 778, "Selected rows", 12, italic=True)

# quantize the residual
text(582, 554, "*Quantize the residual*", 15, ha="left")
text(727, 586, "*4-bit* $q$_{1}", 14)
text(775, 616, "$r$ = $x$ − $ς$_{1}$q$_{1}", 14)
text(1055, 586, "*4-bit* $q$_{2}", 14)
text(1053, 615, "$ς$_{2} ≤ $ς$_{1}/14", 14)
text(592, 652, "$ς$_{1}", 14)
path([(826, 682), (950, 638), (950, 722)], None, fill="F2C9BA", alpha=45, close=True)
rect(950, 638, 210, 84, "FCEBC8", None, radius=5)
line(595, 682, 826, 682, LINK, 2.0)
line(826, 682, 921, 682, "C5DAF3", 1.4)
for x in (595, 643, 692, 826):
    line(x, 673, x, 691, LINK, 2.0)
for x in (874, 921):
    line(x, 674, x, 690, "9CC0EA", 1.6)
line(746, 668, 746, 696, LINK, 4.0)
for x in (746, 787):
    line(x, 634, x, 668, lw=1.0, dash=True)
line(749, 640, 784, 640, lw=1.1, both=True, size="sm")
dot(787, 682, 7)
text(801, 654, "$x$", 15)
text(750, 712, "$ς$_{1}$q$_{1}", 14)
line(960, 682, 1152, 682, "D8432B", 1.5)
for k in range(8):
    line(968 + 24.5 * k, 673, 968 + 24.5 * k, 691, "D8432B", 1.5)
line(1066, 669, 1066, 695, RES_EDGE, 4.0)
text(1068, 710, "$ς$_{2}$q$_{2}", 14)
text(804, 752, "Error bound:  $ς$_{1}/2 →", 13, ha="right", italic=True)
text(810, 752, "*$ς$_{1}/28*", 13.5, RES_EDGE, ha="left", italic=True)
chip(985, 752, 40, 25, "$q$_{2}", RES, RES_EDGE, 13)
text(1036, 754, "$x̂$ = $ς$_{1}$q$_{1} + $ς$_{2}$q$_{2}", 14, ha="left")

# apply the two codes
text(1213, 554, "*Apply 4-bit codes*", 15, ha="left")
text(1497, 548, "$Q$_{4}($W$)", 12.5, LINK, ha="left")
text(1218, 612, "*Linear*", 15, ha="left")
small = [[BLUES[3] if r == c else BLUES[(r + c * 2) % 3] for c in range(4)] for r in range(3)]
for y, label, fill, edge, rows in [(598, "$q$_{1}", BASE, BASE_EDGE, "(All rows)"),
                                   (643, "$q$_{2}", RES, RES_EDGE, "(Selected rows)")]:
    grid(1307, y, 55, 32, small, edge="FFFFFF", frame="1F4E8C")
    text(1385, y + 15, "×", 16)
    chip(1404, y + 2, 35, 27, label, fill, edge, 14)
    text(1447, y + 15, rows, 12.5, ha="left")
brace(1565, 598, 13, 80)
text(1590, 630, "Scaled\nsum", 12.5, ha="left")
text(1218, 717, "*K/V*", 15, ha="left")
text(1345, 695, "$q$_{1}", 14)
text(1398, 695, "$q$_{2}", 14)
for k in range(4):
    y = 708 + 20.5 * k
    rect(1321, y, 104, 20.5, None, RULE, 0.9)
    rect(1334, y + 4, 23, 12.5, BASE, BASE_EDGE, 1.0, radius=2)
    if k in (1, 2):
        rect(1387, y + 4, 23, 12.5, RES, "D8432B", 1.0, radius=2)
brace(1432, 710, 13, 80)
text(1455, 748, "Pair codes\nif selected", 12.5, ha="left", italic=True)
line(1578, 668, 1618, 668)
path([(1545, 750), (1618, 750), (1618, 668)], INK, 1.4, arrow=False)
dot(1618, 668, 4)
line(1618, 668, 1670, 668, arrow=True)
shadow(rect(1672, 635, 90, 66, "E4E4E8", RULE, 1.4, radius=10))
text(1717, 668, "*Generate\naction*", 14)
path([(1717, 701), (1717, 837), (1362, 837)], INK, 1.6)

# the gaze of this control step
text(1240, 803, "Action queries × future keys", 12.5, italic=True)
attention = [[0.72 if c in (1, 4) else 0.03 + 0.03 * ((r + c) % 2) for c in range(7)] for r in range(3)]
grid(1147, 820, 166, 55, [[ramp(HEAT[:3], v) for v in row] for row in attention], edge="E8D9B0")
text(1334, 840, "···", 13)
line(1125, 832, 912, 832, arrow=True)
text(862, 828, "@Σ@", 28)
text(868, 859, "Queries, heads, layers, steps", 12, italic=True)
line(818, 832, 667, 832, arrow=True)
text(637, 835, "···", 13)
text(545, 806, "*Runtime gaze* g_{j}", 13.5, italic=True)
grid(455, 823, 161, 27, [[ramp(HEAT, g) for g in [0.04, 0.08, 0.22, 0.30, 0.66, 0.48]]], edge="B9B9BE")
path([(443, 836), (54, 836), (54, 672)], GAZE, 2.2, dash=True)
text(174, 812, "Next selection", 13.5, GAZE, italic=True)

deck.save(OUT)
print("wrote", OUT)
