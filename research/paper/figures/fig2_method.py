"""Fig. 2, the overview of LAQ, as an editable PowerPoint slide.

Two rows that read left to right: what is computed offline (a-c), a strip with the quantities that are handed
to run time, and the loop of every control step (d). Arrows only join neighbours, so every arrow is short and
straight; the one long line is the gaze that returns to the next control step.

Coordinates are points of the printed figure (495 pt = \\textwidth, origin at the top left). The slide is K times
larger, so that it is comfortable to edit; text of 7 pt here prints at 7 pt.

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
W, H, K = 495.0, 236.0, 2.0

FONT, MATH = "Cambria", "Cambria Math"
INK, MUTED, HAIR = "222222", "6E6E73", "C9CED6"
BLUE = "0F4D92"                        # the reader metric and what follows from it
BASE, BASE_EDGE = "C9DBF2", "3F7BC8"   # first 4-bit pass
RES, RES_EDGE = "F9C66B", "D86A12"     # second 4-bit pass, and the gaze that selects it
FROZEN, FRESH = "B64342", "9A9AA0"
CARD_A, CARD_A_EDGE = "F4F7FB", "D5DDE9"
CARD_B, CARD_B_EDGE = "FFF8EE", "EED9BA"
BAND, BADGE = "DDE4EE", "2E3A4B"
HEAT = ["FFF6D6", "FDCB6E", "F28C38", "D8432B", "8C1020"]
BLUES = ["F2F6FC", "D5E3F6", "A9C6EC", "2F6FC4"]

deck = Presentation()
deck.slide_width, deck.slide_height = Emu(int(W * K * 12700)), Emu(int(H * K * 12700))
shapes = deck.slides.add_slide(deck.slide_layouts[6]).shapes
E = lambda v: Emu(int(round(v * K * 12700)))  # noqa: E731
rgb = RGBColor.from_string


def ramp(colors, t):
    t = min(max(t, 0.0), 1.0) * (len(colors) - 1)
    k = min(int(t), len(colors) - 2)
    a, b = (tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)) for c in (colors[k], colors[k + 1]))
    return "".join(f"{round(a[i] + (b[i] - a[i]) * (t - k)):02X}" for i in range(3))


def runs(markup):
    """`$` toggles italic, `*` bold, `%` underline, `@` the math font, `~` blue, `|` orange; `_{}` and `^{}` set an index."""
    out, state, buf, i = [], dict(italic=False, bold=False, math=False, under=False, tint=None), "", 0

    def flush(base=0, slant=None):
        nonlocal buf
        if buf:
            out.append((buf, state["italic"] if slant is None else slant, state["bold"], state["math"], base, state["tint"], state["under"]))
        buf = ""

    while i < len(markup):
        c = markup[i]
        if c in "$*%@~|":
            flush()
            if c in "~|":
                tint = BLUE if c == "~" else RES_EDGE
                state["tint"] = None if state["tint"] == tint else tint
            else:
                key = {"$": "italic", "*": "bold", "%": "under", "@": "math"}[c]
                state[key] = not state[key]
        elif c in "_^" and markup[i + 1:i + 2] == "{":
            flush()
            end = markup.index("}", i)
            inner = markup[i + 2:end]  # a `$` inside the braces sets the whole index in italic
            buf = inner.replace("$", "")
            flush(-25000 if c == "_" else 30000, state["italic"] or "$" in inner)
            i = end
        else:
            buf += c
        i += 1
    flush()
    return out


def text(x, y, markup, size=7.0, color=INK, ha="center", bold=False, italic=False):
    """Text centred vertically on y; `ha` places it about x. Lines are separated by a newline."""
    rows = markup.split("\n")
    plain = max(len("".join(r[0] for r in runs(row))) for row in rows)
    w, h = plain * size * 0.6 + 6, len(rows) * size * 1.2 + 2
    left = {"center": x - w / 2, "left": x, "right": x - w}[ha]
    shape = shapes.add_textbox(E(left), E(y - h / 2), E(w), E(h))
    frame = shape.text_frame
    frame.word_wrap, frame.auto_size, frame.vertical_anchor = False, MSO_AUTO_SIZE.NONE, MSO_ANCHOR.MIDDLE
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    for k, row in enumerate(rows):
        paragraph = frame.paragraphs[0] if k == 0 else frame.add_paragraph()
        paragraph.alignment = {"center": PP_ALIGN.CENTER, "left": PP_ALIGN.LEFT, "right": PP_ALIGN.RIGHT}[ha]
        paragraph.line_spacing = 0.92
        for s, it, bd, math, base, tint, under in runs(row):
            run = paragraph.add_run()
            run.text = s
            font = run.font
            font.name, font.size = (MATH if math else FONT), Pt(size * K)
            font.italic, font.bold = it ^ italic, bd ^ bold
            font.color.rgb = rgb(tint or color)
            if under:
                font.underline = True
            if base:
                run._r.get_or_add_rPr().set("baseline", str(base))
    return shape


def style(shape, fill=None, edge=None, lw=0.6, alpha=None, dash=False):
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
        shape.line.color.rgb, shape.line.width = rgb(edge), Pt(lw * K)
        if dash:
            shape.line.dash_style = MSO_LINE.DASH
    else:
        shape.line.fill.background()
    return shape


def shadow(shape, blur=2.2, dist=0.9, alpha=22):
    """A soft shadow below the shape; blur and distance in printed points."""
    effects = shape._element.spPr.find(qn("a:effectLst"))
    outer = effects.makeelement(qn("a:outerShdw"), {"blurRad": str(int(blur * K * 12700)), "dist": str(int(dist * K * 12700)),
                                                    "dir": "5400000", "algn": "t", "rotWithShape": "0"})
    colour = outer.makeelement(qn("a:srgbClr"), {"val": "000000"})
    colour.append(colour.makeelement(qn("a:alpha"), {"val": str(alpha * 1000)}))
    outer.append(colour)
    effects.append(outer)
    return shape


def rect(x, y, w, h, fill=None, edge=None, lw=0.6, radius=0.0, into=None, alpha=None):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = (into or shapes).add_shape(kind, E(x), E(y), E(w), E(h))
    if radius:
        shape.adjustments[0] = min(radius / min(w, h), 0.5)
    return style(shape, fill, edge, lw, alpha)


def picture(x, y, w, h, name):
    return shapes.add_picture(str(ASSETS / name), E(x), E(y), E(w), E(h))


def heads(shape, end=True, start=False, size="med"):
    outline = shape.line._get_or_add_ln()
    if start:
        outline.append(outline.makeelement(qn("a:headEnd"), {"type": "triangle", "w": size, "len": size}))
    if end:
        outline.append(outline.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": size, "len": size}))


def line(x1, y1, x2, y2, color=INK, lw=0.8, arrow=False, both=False, dash=False, size="med"):
    shape = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, E(x1), E(y1), E(x2), E(y2))
    shape.line.color.rgb, shape.line.width = rgb(color), Pt(lw * K)
    if dash:
        shape.line.dash_style = MSO_LINE.DASH
    heads(shape, end=arrow or both, start=both, size=size)
    outline = shape.line._get_or_add_ln()
    outline.addnext(outline.makeelement(qn("a:effectLst"), {}))
    return shape


def path(points, color=INK, lw=0.8, arrow=True, dash=False, fill=None, alpha=None, close=False, size="med"):
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


def grid(x, y, w, h, colors, edge="FFFFFF", lw=0.35, frame=INK, frame_lw=0.6):
    """A grid of filled cells (rows of hex colours) in one group, with a frame around it."""
    group = shapes.add_group_shape()
    rows, cols = len(colors), len(colors[0])
    for r in range(rows):
        for c in range(cols):
            rect(x + w * c / cols, y + h * r / rows, w / cols, h / rows, colors[r][c], edge, lw, into=group.shapes)
    if frame:
        rect(x, y, w, h, None, frame, frame_lw, into=group.shapes)
    return group


def dot(x, y, r=1.6, color=INK):
    return style(shapes.add_shape(MSO_SHAPE.OVAL, E(x - r), E(y - r), E(2 * r), E(2 * r)), color)


def chip(x, y, w, h, label, fill, edge, size=7.0):
    shadow(rect(x, y, w, h, fill, edge, 0.7, radius=2.2), 1.4, 0.6, 20)
    text(x + w / 2, y + h / 2, label, size)


def pill(x, y, w, label, size=6.4, fill="FFFFFF", edge=HAIR, color=MUTED, ha="right"):
    """A small tag, 9.4 pt tall, centred vertically on y; x is its right, left or centre edge."""
    left = {"right": x - w, "left": x, "center": x - w / 2}[ha]
    rect(left, y - 4.7, w, 9.4, fill, edge, 0.5, radius=4.7)
    text(left + w / 2, y, label, size, color)


def panel_title(x, y, letter, name):
    rect(x, y - 4.6, 9.2, 9.2, BADGE, None, radius=2.0)
    text(x + 4.6, y - 0.1, f"*{letter}*", 6.8, "FFFFFF")
    text(x + 12.5, y, f"*{name}*", 8.2, ha="left")


def step_title(x, y, number, name):
    style(shapes.add_shape(MSO_SHAPE.OVAL, E(x), E(y - 4.4), E(8.8), E(8.8)), RES_EDGE)
    text(x + 4.4, y - 0.1, f"*{number}*", 6.4, "FFFFFF")
    text(x + 11.5, y, f"*{name}*", 7.5, ha="left")


def stub(x, y1, y2=108.6):
    """An offline quantity dropping into the strip below."""
    line(x, y1, x, y2, INK, 0.75, arrow=True, size="sm")


# ---- the two rows and the strip between them --------------------------------------------------------------
shadow(rect(2, 2, 491, 99, CARD_A, CARD_A_EDGE, 0.6, radius=5), 2.6, 1.0, 14)
shadow(rect(2, 126, 491, 108, CARD_B, CARD_B_EDGE, 0.6, radius=5), 2.6, 1.0, 14)
rect(2, 109, 491, 12.4, BAND, None, radius=3.2)

# ---- (a) reader metric -----------------------------------------------------------------------------------
X, CH = 7.0, 79.0  # CH: the height of the chain  Z_M -> n_u -> allocation
panel_title(X, 10.5, "a", "Reader metric")
text(X + 5.5, 30, "$z$^{∗}", 8.5)
for k in range(5):
    xl = X + 12 + 3.4 * k
    shadow(path([(xl, 24), (xl + 13, 17), (xl + 13, 35), (xl, 42)], "3F7BC8", 0.5, close=True, fill="DCE9FA"), 1.2, 0.5, 16)
line(X + 41.5, 30, X + 50.5, 30, arrow=True, size="sm")
shadow(rect(X + 52, 22.5, 44, 15, "E6E8EC", "9AA1AC", 0.7, radius=3.5), 1.6, 0.7, 22)
text(X + 74, 30, "*Reader* $R$", 7.4)
line(X + 97.5, 30, X + 106.5, 30, arrow=True, size="sm")
for k in range(3):
    shadow(rect(X + 108.5 + 8.6 * k, 26, 7, 8, "CFC8EA", "6B6799", 0.5, radius=1), 1.0, 0.4, 16)
text(X + 141.5, 30, "$a$^{∗}", 8.5)
path(curve((X + 74, 38), (X + 68, 50.5), (X + 31, 44.5)), BLUE, 0.8, size="sm")
text(X + 74.5, 44.05, "%*reader probes*%", 6.0, BLUE, ha="left", italic=True)

THIN = " "
text(X + 3, 56, f"$M$ = E_{{$c$}}[$J$_{{$c$}}^{{$T$}}{THIN}$D$_{{$c$}}{THIN}$J$_{{$c$}}]", 7.2, ha="left")
grid(X + 10, 62, 34, 34, [[BLUES[3] if r == c else BLUES[(r * 3 + c * 5) % 3] for c in range(6)] for r in range(6)])
stub(X + 27, 96.6)
line(X + 47, CH, X + 77, CH, arrow=True, size="sm")
text(X + 62, 73.6, "tr($M$_{$jj$})", 6.8)
text(X + 62, 84.6, "per token", 6.3, MUTED, italic=True)
text(X + 97, 56.3, "*token sensitivity*", 6.6)
picture(X + 79, 61, 36, 36, "b_frame_1.png")     # third-person view of a predicted LIBERO frame, in grey
picture(X + 79, 61, 36, 36, "b_overlay_1.png")   # share of tr(M) per token, as in Fig. 3(b)
rect(X + 79, 61, 36, 36, None, INK, 0.6)
text(X + 134.5, 68.5, "future error", 6.3, MUTED, italic=True)
text(X + 130, CH, "~*$Z$_{$M$}*~", 9.5)

# ---- (b) read count --------------------------------------------------------------------------------------
X = 171.0
line(149, CH, X + 35, CH, arrow=True)           # Z_M enters the measurement
panel_title(X, 10.5, "b", "Read count")
STEP, DOTS, SUM = [X + 69, X + 84, X + 110], X + 97, X + 125
text(X + 62, 20.5, "denoising step", 6.3, MUTED, ha="right", italic=True)
for x, label in zip(STEP + [DOTS], ["1", "2", "$T$", "···"]):
    text(x, 20.5, label, 6.8)
text(SUM + 12.5, 20.5, "sum", 6.3, MUTED, italic=True)
line(X + 119.5, 16.5, X + 119.5, 67.5, "E3E6EB", 0.5)
for y, colour in ((25.5, HAIR), (46.5, "E3E6EB"), (67.5, HAIR)):
    line(X, y, X + 150, y, colour, 0.5)
text(X, 33, "*Frozen*", 7.6, FROZEN, ha="left")
for x in STEP:
    line(x - 5.5, 33, x + 5.5, 33, FROZEN, 1.0, arrow=True, size="sm")
text(DOTS, 32.5, "···", 6.8, FROZEN)
line(SUM, 33, SUM + 25, 33, FROZEN, 1.7, arrow=True, size="sm")
text(X, 41, "weights, caches, static tokens", 6.3, MUTED, ha="left", italic=True)
text(X, 54, "*Fresh*", 7.6, MUTED, ha="left")
for x, (dx, dy) in zip(STEP, [(5.5, 0), (3.9, -3.9), (-3.9, -3.9)]):
    line(x - dx, 54 - dy, x + dx, 54 + dy, FRESH, 1.0, arrow=True, size="sm")
text(DOTS, 53.5, "···", 6.8, FRESH)
line(SUM, 54, SUM + 8, 54, FRESH, 1.7, arrow=True, size="sm")
text(X, 62, "activations of tokens being denoised", 6.3, MUTED, ha="left", italic=True)

text(X + 41, CH, "$n$_{$u$} =", 8.2, ha="left")
text(X + 85.5, CH - 5.0, "~$Z$_{$M$}~(frozen)", 7.4)
line(X + 61, CH, X + 110, CH, lw=0.7)
text(X + 85.5, CH + 5.4, "~$Z$_{$M$}~(fresh)", 7.4)
stub(X + 86, CH + 10.4)
text(X + 150, 93.5, "fresh sources:  n_{u} = 1", 6.3, MUTED, ha="right", italic=True)

# ---- (c) joint allocation --------------------------------------------------------------------------------
X = 331.0
line(X - 47, CH, X + 4.5, CH, arrow=True)       # n_u enters the score
text(X - 21, CH - 5.4, "$n$_{$u$}", 7.4)
panel_title(X, 10.5, "c", "Joint allocation")
pill(488, 10.5, 49, "Offline, once")
text(X + 78, 27.2, f"@ℒ@ = E_{{$c$}}[$A$ + $λ$_{{$M$}}{THIN}~$Z$_{{$M$}}~ + $βK$]", 8.6)
for x0, x1, name in [(X + 63.2, X + 68.2, "action"), (X + 76.7, X + 95.7, "future"), (X + 104.2, X + 113.9, "reader")]:
    path([(x0, 22.9), (x0, 21.9), (x1, 21.9), (x1, 22.9)], MUTED, 0.4, arrow=False)
    text((x0 + x1) / 2, 18.2, name, 6.0, MUTED, italic=True)
path([(X + 78, 32.8), (X + 78, 36.4), (X + 42, 36.4), (X + 42, 41.6)], INK, 0.75, size="sm")
path([(X + 78, 32.8), (X + 78, 36.4), (X + 124, 36.4), (X + 124, 41.6)], INK, 0.75, size="sm")
text(X + 42, 46.4, "*Activations and K/V*", 7.2)
text(X + 124, 46.4, "*Weights*", 7.2)
line(X + 6, 51, X + 6, 90, lw=0.75)
line(X + 6, 90, X + 80, 90, lw=0.75)
for k, top in enumerate([54, 61, 67, 75, 80, 85]):
    bar = rect(X + 10 + 11.6 * k, top, 8.6, 90 - top, ["7E828C", "8E929C", "A2A6AF"][min(k, 2)], "6B7280", 0.4)
    if k % 2:  # cached keys and values
        bar.fill.patterned()
        bar.fill.pattern = MSO_PATTERN.WIDE_UPWARD_DIAGONAL
        bar.fill.fore_color.rgb, bar.fill.back_color.rgb = rgb("7E828C"), rgb("FFFFFF")
line(X + 6.5, 71, X + 81, 71, lw=0.7, dash=True)
text(X + 86, 70.6, "$μ$", 8.5)
rect(X + 50, 52.5, 5, 5, "8E929C", "6B7280", 0.4)
text(X + 57.5, 55, "activation", 6.2, MUTED, ha="left")
key = rect(X + 50, 60.5, 5, 5, None, "6B7280", 0.4)
key.fill.patterned()
key.fill.pattern = MSO_PATTERN.WIDE_UPWARD_DIAGONAL
key.fill.fore_color.rgb, key.fill.back_color.rgb = rgb("7E828C"), rgb("FFFFFF")
text(X + 57.5, 63, "K/V", 6.2, MUTED, ha="left")
text(X + 43, 95, "budget B → one threshold μ", 6.3, MUTED, italic=True)
stub(X + 91, 74.5)
weights = [[BLUES[3] if r == c else BLUES[(r * 2 + c * 3) % 3] for c in range(4)] for r in range(4)]
grid(X + 101, 53, 17, 17, weights, frame=BLUE)
grid(X + 133, 53, 17, 17, weights, frame=BLUE)
line(X + 120, 61.5, X + 131, 61.5, arrow=True, size="sm")
text(X + 109.5, 75.4, "$W$", 7.2)
text(X + 141.5, 75.6, "$Q$_{4}($W$)", 7.2)
text(X + 121.5, 88.5, "second-order\nrounding", 6.3, MUTED, italic=True)
stub(X + 141.5, 80.4)

# ---- what the offline stage hands to run time ------------------------------------------------------------
for x, w, label in [(39, 66, "first gaze  tr($M$_{$jj$})"), (257, 54, "read count  $n$_{$u$}"),
                    (422, 46, "threshold  $μ$"), (472.5, 32, "$Q$_{4}($W$)")]:
    rect(x - w / 2, 110.6, w, 9.2, "FFFFFF", None, radius=4.6)
    text(x, 115.1, label, 6.6)
    path([(x - 3.4, 122.2), (x + 3.4, 122.2), (x, 125.6)], None, fill="7C8AA0", close=True)

# ---- (d) gaze-guided refinement --------------------------------------------------------------------------
panel_title(7, 135, "d", "Gaze-guided refinement")
pill(488, 135, 71, "At every control step")
pill(262, 135, 124, "*All weights 4-bit, no 16-bit branch*", 6.5, "FFFFFF", CARD_B_EDGE, INK, ha="center")
TOP, MID, STRIP = 147.5, 181.0, 200.0  # step titles, the chain of the four steps, the two gaze strips

# 1: score and select
X = 7.0
step_title(X, TOP, 1, "Score and select")
text(X + 61, 158.4, f"$score$_{{$u$}} = $n$_{{$u$}}{THIN}$ω$_{{$u$}}{THIN}$σ$_{{$u$}}^{{2}} / $c$_{{$u$}}  ≥  $μ$", 7.3)
gaze = [0.05, 0.10, 0.25, 0.38, 0.62, 0.50]
profile = [0.35, 0.75, 1.0, 0.5]
omega = [[min(1.0, 1.45 * p * g) for g in gaze] for p in profile]
text(X + 10.5, 181, "$φ$_{$s$}", 7.8)
grid(X + 20, 165, 8.4, 32, [[ramp(BLUES[:3], p)] for p in profile], edge="FFFFFF", frame="6B7280")
grid(X + 32, 165, 54, 32, [[ramp(HEAT, v) for v in row] for row in omega], edge="FFFFFF", frame=INK)
for r, row in enumerate(omega):
    for c, v in enumerate(row):
        if v >= 0.5:  # the tokens and steps above the threshold
            rect(X + 32 + 9 * c + 0.5, 165 + 8 * r + 0.5, 8, 7, None, BADGE, 0.9)
text(X + 99, 172.5, "$ω$_{$sj$}", 7.8)
rect(X + 90.5, 183.4, 6.4, 5.6, None, BADGE, 0.9)
text(X + 99.5, 186.3, "selected", 6.3, MUTED, ha="left", italic=True)
grid(X + 32, STRIP, 54, 8, [[ramp(HEAT, g) for g in gaze]], edge="FFFFFF", frame=INK)
text(X + 29.5, STRIP + 4, "gaze $g$_{$j$}", 6.8, ha="right")
line(X + 123.5, MID, X + 130.6, MID, INK, 1.2, arrow=True)

# 2: quantize the residual
X = 138.0
C, CELL = X + 68, 15.5  # the coarse level nearest to x, and the coarse step
step_title(X, TOP, 2, "Quantize the residual")
text(X + 1, 158.4, "~*1st pass:* 4-bit $q$_{1}~", 6.9, ha="left")
text(X + 135, 158.4, "$x̂$ = $ς$_{1}$q$_{1} + $ς$_{2}$q$_{2}", 7.3, ha="right")
path([(C - CELL / 2, 171.5), (C + CELL / 2, 171.5), (X + 126, 186.5), (X + 10, 186.5)], None, fill="FBDDB0", alpha=55, close=True)
rect(X + 10, 186.5, 116, 19, "FCEBC8", None, radius=2)
line(X + 6, 171.5, X + 130, 171.5, BASE_EDGE, 1.0)
for k in range(9):
    line(X + 6 + CELL * k, 168.3, X + 6 + CELL * k, 174.7, BASE_EDGE, 1.0)
line(C, 166.6, C, 176.4, BLUE, 2.0)
dot(C + 5.3, 171.5, 1.9)
text(C - 8.5, 165.4, "$ς$_{1}$q$_{1}", 6.8, BLUE)
text(C + 8.2, 165.4, "$x$", 7.4)
text(C, 180.4, "$r$ = $x$ − $ς$_{1}$q$_{1}", 6.6)
line(X + 13, 197, X + 123, 197, RES_EDGE, 0.8)
for k in range(15):
    line(X + 17.25 + 7.25 * k, 194.2, X + 17.25 + 7.25 * k, 199.8, RES_EDGE, 0.8)
line(C, 193.0, C, 201.0, BLUE, 1.3)
line(X + 104.25, 192.4, X + 104.25, 201.6, RES_EDGE, 2.0)
dot(X + 103.1, 197, 1.9)
line(C + 1.0, 190.2, X + 102.8, 190.2, INK, 0.5, both=True, size="sm")
text(X + 113.5, 202.6, "$ς$_{2}$q$_{2}", 6.6, RES_EDGE)
text(X + 1, 211.6, "|*2nd pass:* 4-bit $q$_{2}|", 6.9, ha="left")
text(X + 135, 211.6, "$error:  ς$_{1}/2 → |*$ς$_{1}/28*|", 6.9, ha="right")
line(X + 137.4, MID, X + 144.4, MID, INK, 1.2, arrow=True)

# 3: apply both codes
X = 283.0
step_title(X, TOP, 3, "Apply both codes")
small = [[BLUES[3] if r == c else BLUES[(r + c * 2) % 3] for c in range(4)] for r in range(3)]
text(X, 162.5, "*Linear*", 7.0, ha="left")
for x, label, fill, edge, note in [(X + 27, "$q$_{1}", BASE, BASE_EDGE, "all rows"), (X + 72, "$q$_{2}", RES, RES_EDGE, "selected rows")]:
    grid(x, 157, 15, 11, small, frame=BLUE)
    text(x + 19.2, 162.3, "×", 7.2)
    chip(x + 23, 157.3, 13, 10.4, label, fill, edge, 7.0)
    text(x + 18, 173.4, note, 6.3, MUTED, italic=True)
text(X + 67.3, 162.3, "+", 7.6)
text(X, 190, "*K/V*", 7.0, ha="left")
for k in range(5):
    rect(X + 27 + 10 * k, 181, 9, 8, BASE, BASE_EDGE, 0.5, radius=1)
    if k in (2, 3):
        rect(X + 27 + 10 * k, 190.6, 9, 8, RES, RES_EDGE, 0.5, radius=1)
    else:
        rect(X + 27 + 10 * k, 190.6, 9, 8, None, "DCCFB6", 0.4, radius=1)
text(X + 81, 185, "$q$_{1}", 7.0, ha="left")
text(X + 81, 194.6, "$q$_{2}", 7.0, ha="left")
text(X + 55, 206.6, "second code stored next to the first", 6.3, MUTED, italic=True)
line(X + 111.4, MID, X + 118.4, MID, INK, 1.2, arrow=True)

# 4: update the gaze
X = 402.0
step_title(X, TOP, 4, "Update the gaze")
text(X + 43, 158.4, "action-token attention", 6.3, MUTED, italic=True)
new_gaze = [0.04, 0.08, 0.22, 0.30, 0.66, 0.48]
attention = [[min(1.0, g * (0.75 + 0.5 * ((r * 2 + c) % 3) / 2)) for c, g in enumerate(new_gaze)] for r in range(3)]
grid(X + 30, 164, 54, 21, [[ramp(HEAT, v) for v in row] for row in attention], edge="FFFFFF", frame=INK)
text(X + 27.5, 174.5, "action\ntokens", 6.3, MUTED, ha="right", italic=True)
line(X + 57, 186.2, X + 57, 198.6, INK, 0.75, arrow=True, size="sm")
text(X + 53.5, 192.2, "@Σ@ layers, heads, steps", 6.3, MUTED, ha="right")
grid(X + 30, STRIP, 54, 8, [[ramp(HEAT, g) for g in new_gaze]], edge="FFFFFF", frame=INK)
text(X + 27.5, STRIP + 4, "gaze $g$_{$j$}", 6.8, ha="right")

# the gaze of this control step selects the tokens of the next
path([(X + 57, STRIP + 8.6), (X + 57, 226.5), (66, 226.5), (66, STRIP + 9.0)], RES_EDGE, 1.0, dash=True, size="sm")
text(262, 220.6, "gaze for the next control step", 6.8, RES_EDGE, italic=True)

deck.save(OUT)
print("wrote", OUT)
