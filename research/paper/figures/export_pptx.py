"""Export the first slide of a PowerPoint file through PowerPoint itself: a vector PDF and, optionally, a PNG.

    python research/paper/figures/export_pptx.py deck.pptx out.pdf [preview.png]

The deck is copied first, so it can stay open in PowerPoint while this runs.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import win32com.client

source, pdf = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
copy = Path(tempfile.mkdtemp()) / source.name
shutil.copy(source, copy)
app = win32com.client.Dispatch("PowerPoint.Application")
deck = app.Presentations.Open(str(copy), True, False, False)
deck.SaveAs(str(pdf), 32)  # ppSaveAsPDF
if len(sys.argv) > 3:
    width = 3600
    deck.Slides(1).Export(str(Path(sys.argv[3]).resolve()), "PNG", width, round(width * deck.PageSetup.SlideHeight / deck.PageSetup.SlideWidth))
deck.Close()
print("wrote", pdf)
