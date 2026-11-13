"""Set the typeface of every text run of a PowerPoint file, in place, and optionally scale the text.

    python research/paper/figures/set_pptx_font.py deck.pptx [Cambria] [scale]

The figures of the paper are edited by hand in PowerPoint, so the font is changed in the slides themselves.
"""
import sys
from pathlib import Path

from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Pt

path = Path(sys.argv[1])
font = sys.argv[2] if len(sys.argv) > 2 else "Cambria"
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

deck = Presentation(path)
count = 0
for slide in deck.slides:
    for properties in slide._element.iter(qn("a:rPr"), qn("a:endParaRPr"), qn("a:defRPr")):
        for tag in ("a:latin", "a:ea", "a:cs"):
            for node in properties.findall(qn(tag)):
                properties.remove(node)
    for run in slide._element.iter(qn("a:r")):
        properties = run.find(qn("a:rPr"))
        if properties is None:
            properties = run.makeelement(qn("a:rPr"), {})
            run.insert(0, properties)
        latin = properties.makeelement(qn("a:latin"), {"typeface": font})
        anchor = [properties.find(qn(tag)) for tag in ("a:sym", "a:hlinkClick", "a:hlinkMouseOver", "a:rtl", "a:extLst")]
        anchor = next((node for node in anchor if node is not None), None)
        if anchor is not None:
            anchor.addprevious(latin)
        else:
            properties.append(latin)
        if scale != 1.0 and properties.get("sz"):
            properties.set("sz", str(round(int(properties.get("sz")) * scale)))
        count += 1
deck.save(path)
print(f"{path.name}: {count} runs in {font}" + (f", sizes x{scale}" if scale != 1.0 else ""))
