"""One-off edit of CVPR2027/figures/figure1_lavq_editable.pptx, which is edited by hand and is the source of
figure1_a.pdf: recolours panel (a) to the palette of Fig. 2, so that a colour means the same thing in every figure.

  blue    the predicted future at full precision: in the BF16 row and where it is swapped in (was orange and purple)
  red     everything 4-bit and the error it causes
  slate   the BF16 modules
  orange  the conclusion only: the reduction of the action error and the sentence below it

Run once, then export with export_fig1a.py. Panels (b) and (c) of the slide are not in the paper and are left alone.
"""
from pathlib import Path

from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Inches

PATH = Path(__file__).resolve().parents[3] / "CVPR2027/figures/figure1_lavq_editable.pptx"
PANEL_A = Inches(10.02)  # shapes of panel (a) start left of this

PALETTE = {
    # ink and neutrals
    "111820": "222222", "5F6C7A": "6E6E73", "657786": "6E6E73", "6B7480": "6E6E73", "99A5AF": "9A9AA0",
    # the card
    "F2F8FD": "F4F7FB", "6B8AA1": "D5DDE9",
    # BF16 modules and the gripper
    "536C82": "6B7280", "667F91": "6B7280", "9BAFC0": "AAB2BF", "EEF2F6": "F0F2F5", "EDF3F6": "F0F2F5",
    "DFE8EF": "E1E5EC", "E3EDF5": "E6EBF3",
    # 4-bit modules, the degraded future and the action error
    "C24A3E": "B64342", "D78073": "D28A8A", "F7E4E1": "F5E5E5", "F6E8E5": "F6E8E8", "FFF1ED": "FBF0F0",
    "F7D5C9": "F0D2D1", "FCE8E2": "F8E6E6",
    # the conclusion, and the object in the gripper
    "CF6D0A": "D86A12", "D9822B": "D86A12", "FCE8D2": "FCEBC8", "F4B565": "F9C66B", "FFF1DA": "FFEDC1",
    # the swap
    "7A5BA6": "3F7BC8",
}
FUTURE = {"D9822B": "3F7BC8", "FBEBD8": "DCE9FA", "7A5BA6": "3F7BC8", "ECE5F3": "DCE9FA"}  # frames of a BF16 future
TEXT = {"7A5BA6": "0F4D92"}
TEXT_TAGS = {qn("a:rPr"), qn("a:defRPr"), qn("a:endParaRPr")}

deck = Presentation(PATH)
shapes = [s for s in deck.slides[0].shapes if s.left < PANEL_A]
assert any(c.get("val") == "7A5BA6" for s in shapes for c in s._element.iter(qn("a:srgbClr"))), "already recoloured"
count = 0
for shape in shapes:
    future = any(key in shape.name for key in ("future frame", "future latent", "legend restored future"))
    for colour in shape._element.iter(qn("a:srgbClr")):
        if colour.getparent().tag == qn("a:outerShdw"):
            continue
        old = colour.get("val").upper()
        text = colour.getparent().getparent().tag in TEXT_TAGS
        new = (FUTURE.get(old) if future else None) or (TEXT.get(old) if text else None) or PALETTE.get(old)
        if new:
            colour.set("val", new)
            count += 1
    if shape.name.startswith("A legend label restored future"):  # the same frames stand in the BF16 row
        runs = [run for paragraph in shape.text_frame.paragraphs for run in paragraph.runs]
        runs[0].text = "BF16 future"
        for run in runs[1:]:
            run.text = ""
deck.save(PATH)
print(f"recoloured {count} colours in {PATH}")
