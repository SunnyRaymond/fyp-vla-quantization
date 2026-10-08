from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET
root = Path('D:/Downloads/Final Year Project')
src = root / 'Weekly Slides/2026-10-07_WAM_Quantization_Progress.pptx'
dst = root / '.codex-tmp/meeting-20261007-revision/source-no-old-tables.pptx'
ns = {'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
      'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
with ZipFile(src) as original, ZipFile(dst, 'w', ZIP_DEFLATED) as edited:
    for info in original.infolist():
        data = original.read(info.filename)
        if info.filename == 'ppt/slides/slide6.xml':
            xml = ET.fromstring(data)
            tree = xml.find('p:cSld/p:spTree', ns)
            for frame in list(tree):
                if frame.find('.//a:tbl', ns) is not None:
                    tree.remove(frame)
            data = ET.tostring(xml, encoding='utf-8', xml_declaration=True)
        edited.writestr(info, data)
