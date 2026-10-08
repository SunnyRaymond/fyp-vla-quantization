"""Preserve the user's deck; transplant only the two authored slide parts."""
import zipfile, pathlib, posixpath, copy, re, xml.etree.ElementTree as ET
ROOT=pathlib.Path(r'D:/Downloads/Final Year Project')
TMP=ROOT/'.codex-tmp/meeting-20261007-strict'
SOURCE=ROOT/'Weekly Slides/2026-10-07_WAM_Quantization_Progress_Revised.pptx'
P='http://schemas.openxmlformats.org/presentationml/2006/main'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
R='http://schemas.openxmlformats.org/package/2006/relationships'
CT='http://schemas.openxmlformats.org/package/2006/content-types'
ET.register_namespace('p',P);ET.register_namespace('a',A);ET.register_namespace('r','http://schemas.openxmlformats.org/officeDocument/2006/relationships')
with zipfile.ZipFile(SOURCE) as src, zipfile.ZipFile(TMP/'authored.pptx') as authored:
    parts={i.filename:src.read(i.filename) for i in src.infolist()}
    protected={n:parts[f'ppt/slides/slide{n}.xml'] for n in [*range(1,10),12]}
    for n,asset in [(10,'imagegen-method-reference.png'),(11,'imagegen-geometry-reference.png')]:
        slide_path=f'ppt/slides/slide{n}.xml'
        rel_path=f'ppt/slides/_rels/slide{n}.xml.rels'
        original_rels=ET.fromstring(src.read(rel_path));new_rels=ET.fromstring(authored.read(rel_path))
        original_targets={r.attrib['Type']:r.attrib['Target'] for r in original_rels}
        for rel in new_rels:
            kind=rel.attrib['Type'].rsplit('/',1)[-1]
            if kind=='image':
                candidate_target=posixpath.normpath(posixpath.join('ppt/slides',rel.attrib['Target'])).lstrip('/')
                media=f'ppt/media/strict-reference-{n}.png'
                assert media not in parts
                data=authored.read(candidate_target)
                assert data==(ROOT/'Weekly Slides/2026-10-07_Figure_Designs'/asset).read_bytes(), 'Reference image bytes changed'
                parts[media]=data;rel.attrib['Target']=f'../media/strict-reference-{n}.png'
            elif kind=='notesSlide':
                candidate_note=posixpath.normpath(posixpath.join('ppt/slides',rel.attrib['Target'])).lstrip('/')
                rel.attrib['Target']=original_targets[rel.attrib['Type']]
                original_note=posixpath.normpath(posixpath.join('ppt/slides',rel.attrib['Target']))
                parts[original_note]=authored.read(candidate_note)
            elif kind=='slideLayout':
                rel.attrib['Target']=original_targets[rel.attrib['Type']]
            else:
                raise RuntimeError('Unexpected relationship: '+kind)
        tree=ET.fromstring(authored.read(slide_path));sp=tree.find(f'{{{P}}}cSld/{{{P}}}spTree')
        editable=[];reference=None
        for el in list(sp):
            if el.tag in [f'{{{P}}}nvGrpSpPr',f'{{{P}}}grpSpPr']:continue
            name=el.find(f'.//{{{P}}}cNvPr')
            if el.tag==f'{{{P}}}pic':
                reference=el;name.attrib['name']='原生成图（移开以编辑下层）';name.attrib['descr']='原生成图完整显示，保持原图比例。选中移开此图，可编辑下方重建元素。'
            elif name is None or name.attrib.get('name') not in ['Title 1','矩形 2','Slide number']:
                editable.append(el)
        assert reference is None and len(editable)>100, 'The two diagrams must use individual native elements, without raster overlays.'
        # Native text runs keep codeword indices editable as actual subscripts.
        for paragraph in tree.findall(f'.//{{{A}}}p'):
            for run in list(paragraph):
                t=run.find(f'{{{A}}}t')
                if t is None or 'c_z' not in (t.text or ''):continue
                index=list(paragraph).index(run);paragraph.remove(run)
                pieces=re.split(r'(c_z)',t.text)
                for piece in pieces:
                    if not piece:continue
                    tokens=[('c',False),('z',True)] if piece=='c_z' else [(piece,False)]
                    for value,subscript in tokens:
                        nr=copy.deepcopy(run);nr.find(f'{{{A}}}t').text=value
                        if subscript:
                            props=nr.find(f'{{{A}}}rPr')
                            if props is None:props=ET.Element(f'{{{A}}}rPr');nr.insert(0,props)
                            props.attrib['baseline']='-25000'
                            if 'sz' in props.attrib:props.attrib['sz']=str(round(int(props.attrib['sz'])*.7))
                        paragraph.insert(index,nr);index+=1
        parts[slide_path]=ET.tostring(tree,encoding='utf-8',xml_declaration=True)
        parts[rel_path]=ET.tostring(new_rels,encoding='utf-8',xml_declaration=True)
    types=ET.fromstring(parts['[Content_Types].xml'])
    if not any(c.attrib.get('Extension')=='png' for c in types):
        ET.SubElement(types,f'{{{CT}}}Default',{'Extension':'png','ContentType':'image/png'})
        parts['[Content_Types].xml']=ET.tostring(types,encoding='utf-8',xml_declaration=True)
    for n,data in protected.items():assert parts[f'ppt/slides/slide{n}.xml']==data
    with zipfile.ZipFile(TMP/'candidate.pptx','w',compression=zipfile.ZIP_DEFLATED) as output:
        for name,data in parts.items():output.writestr(name,data)
    print('12 slides retained; only slides 10 and 11 and their notes were updated; all diagram elements are native and individually editable.')
