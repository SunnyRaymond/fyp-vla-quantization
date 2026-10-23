"""Apply reviewed native figures to the current deck without rebuilding other slides."""
import zipfile,pathlib,posixpath,copy,re,xml.etree.ElementTree as E
TMP=pathlib.Path(r'D:/Downloads/Final Year Project/.codex-tmp/meeting-20261021-review')
P='http://schemas.openxmlformats.org/presentationml/2006/main'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
E.register_namespace('p',P);E.register_namespace('a',A);E.register_namespace('r',R)
def q(ns,t):return '{'+ns+'}'+t
def slide_paths(z):
    rel=E.fromstring(z.read('ppt/_rels/presentation.xml.rels'))
    targets={r.attrib['Id']:posixpath.normpath(posixpath.join('ppt',r.attrib['Target'])).lstrip('/') for r in rel}
    return [targets[s.attrib[q(R,'id')]] for s in E.fromstring(z.read('ppt/presentation.xml')).find(q(P,'sldIdLst'))]
def font(tree):
    for tag in ['rPr','defRPr','endParaRPr']:
        for props in tree.findall('.//'+q(A,tag)):
            for family in ['latin','ea','cs']:
                item=props.find(q(A,family))
                if item is None:item=E.SubElement(props,q(A,family))
                item.set('typeface','Times New Roman')
def operators(tree):
    pattern=re.compile(r'\b(?:NMSE|MSE|KL|mean|max|clamp|round|exp)\b')
    for paragraph in tree.findall('.//'+q(A,'p')):
        for run in list(paragraph):
            node=run.find(q(A,'t'));pr=run.find(q(A,'rPr'))
            if node is None or not node.text or pr is None or pr.get('i')!='1' or not pattern.search(node.text):continue
            pieces=[];start=0
            for m in pattern.finditer(node.text):
                if m.start()>start:pieces.append((node.text[start:m.start()],False))
                pieces.append((m.group(),True));start=m.end()
            if start<len(node.text):pieces.append((node.text[start:],False))
            index=list(paragraph).index(run);paragraph.remove(run)
            for value,upright in pieces:
                child=copy.deepcopy(run);child.find(q(A,'t')).text=value
                if upright:child.find(q(A,'rPr')).set('i','0')
                paragraph.insert(index,child);index+=1
def scripts(tree):
    pattern=re.compile(r'\^\{([^}]+)\}|\^([A-Za-z0-9])|_\{([^}]+)\}|_([A-Za-z0-9]+(?:,[a-z])?)')
    for paragraph in tree.findall('.//'+q(A,'p')):
        for run in list(paragraph):
            node=run.find(q(A,'t'));props=run.find(q(A,'rPr'))
            if node is None or not node.text or props is None or not pattern.search(node.text):continue
            raw=node.text;pieces=[];start=0
            for m in pattern.finditer(raw):
                if m.start()>start:pieces.append((raw[start:m.start()],None))
                pieces.append((m.group(1)or m.group(2)or m.group(3)or m.group(4),'sup'if m.group(1)or m.group(2)else'sub'));start=m.end()
            if start<len(raw):pieces.append((raw[start:],None))
            index=list(paragraph).index(run);paragraph.remove(run)
            for value,sub in pieces:
                r=copy.deepcopy(run);r.find(q(A,'t')).text=value;pr=r.find(q(A,'rPr'))
                if sub:
                    pr.set('baseline','27000'if sub=='sup'else'-23000')
                    if 'sz'in pr.attrib:pr.set('sz',str(round(int(pr.get('sz'))*.73)))
                paragraph.insert(index,r);index+=1
            if '^'in raw or '_{'in raw:
                parent=next((sp for sp in tree.findall('.//'+q(P,'sp'))if paragraph in list(sp.iter())),None)
                if parent is not None:
                    fit=parent.find('.//'+q(A,'normAutofit'))
                    if fit is not None:fit.set('fontScale','100000')
def note_path(z,slide):
    rp=posixpath.join(posixpath.dirname(slide),'_rels',posixpath.basename(slide)+'.rels')
    for r in E.fromstring(z.read(rp)):
        if r.attrib['Type'].endswith('/notesSlide'):return posixpath.normpath(posixpath.join(posixpath.dirname(slide),r.attrib['Target'])).lstrip('/')
def save(parts,path,tree):parts[path]=E.tostring(tree,encoding='utf-8',xml_declaration=True)
with zipfile.ZipFile(TMP/'source-current.pptx')as source,zipfile.ZipFile(TMP/'authored.pptx')as authored:
    parts={i.filename:source.read(i.filename)for i in source.infolist()};old=slide_paths(source);new=slide_paths(authored)
    assert len(old)==18 and len(new)==2
    reviewed={}
    for old_index,new_index in [(2,1),(9,2)]:
        path=old[old_index-1];tree=E.fromstring(parts[path]);made=E.fromstring(authored.read(new[new_index-1]))
        orig=tree.find(q(P,'cSld')+'/'+q(P,'spTree'));made_tree=made.find(q(P,'cSld')+'/'+q(P,'spTree'))
        assert not made_tree.findall(q(P,'pic'))
        for e in list(orig):orig.remove(e)
        for e in list(made_tree):orig.append(copy.deepcopy(e))
        scripts(tree);operators(tree);font(tree);reviewed[old_index]=tree
        save(parts,path,tree)
        on=note_path(source,path);an=note_path(authored,new[new_index-1])
        if on and an:parts[on]=authored.read(an)
    # All figure copies use the same corrected editable objects and font.
    for index,main in [(17,2),(18,9)]:
        path=old[index-1];tree=E.fromstring(parts[path]);st=tree.find(q(P,'cSld')+'/'+q(P,'spTree'));pic=st.find(q(P,'pic'))
        off=pic.find('.//'+q(A,'xfrm')+'/'+q(A,'off'));ext=pic.find('.//'+q(A,'xfrm')+'/'+q(A,'ext'))
        grp=E.Element(q(P,'grpSp'));nv=E.SubElement(grp,q(P,'nvGrpSpPr'))
        E.SubElement(nv,q(P,'cNvPr'),{'id':'3000','name':'Reviewed editable figure, Times New Roman'})
        E.SubElement(nv,q(P,'cNvGrpSpPr'));E.SubElement(nv,q(P,'nvPr'))
        pr=E.SubElement(grp,q(P,'grpSpPr'));xf=E.SubElement(pr,q(A,'xfrm'))
        E.SubElement(xf,q(A,'off'),dict(off.attrib));E.SubElement(xf,q(A,'ext'),dict(ext.attrib))
        E.SubElement(xf,q(A,'chOff'),{'x':str(27*9525),'y':'0'})
        E.SubElement(xf,q(A,'chExt'),{'cx':str(906*9525),'cy':str(round(941*906/1672*9525))})
        for e in reviewed[main].find(q(P,'cSld')+'/'+q(P,'spTree')):
            if e.tag not in [q(P,'nvGrpSpPr'),q(P,'grpSpPr')]:grp.append(copy.deepcopy(e))
        # Group transforms resize shape geometry; PowerPoint needs font sizes scaled explicitly.
        scale=int(ext.get('cx'))/(906*9525)
        for prtag in ['rPr','defRPr','endParaRPr']:
            for props in grp.findall('.//'+q(A,prtag)):
                if 'sz' in props.attrib:props.set('sz',str(round(int(props.get('sz'))*scale)))
        where=list(st).index(pic);st.remove(pic);st.insert(where,grp)
        for counter,nvpr in enumerate(tree.findall('.//'+q(P,'cNvPr')),1):nvpr.set('id',str(counter))
        for t in tree.findall('.//'+q(A,'t')):
            if t.text=='Appendix: ImageGen method design':t.text='Appendix: method figure'
            elif t.text=='Appendix: ImageGen loss design':t.text='Appendix: loss figure'
            elif t.text and t.text.startswith('Generated design reference;'):
                t.text=f'Editable figure, synchronized with slide {main}. All figure text uses Times New Roman.'
        save(parts,path,tree)
    path=old[5];tree=E.fromstring(parts[path])
    for sp in tree.findall('.//'+q(P,'sp')):
        raw=''.join(t.text or ''for t in sp.findall('.//'+q(A,'t')))
        if raw.startswith(('Future donor','Fixed BF16 reader'))or raw=='a':font(sp)
    save(parts,path,tree)
    path=old[7];tree=E.fromstring(parts[path])
    for sp in tree.findall('.//'+q(P,'sp')):
        raw=''.join(t.text or ''for t in sp.findall('.//'+q(A,'t')))
        if raw.startswith('Original objective:'):scripts(sp);font(sp)
    for tc in tree.findall('.//'+q(A,'tc')):
        raw=''.join(t.text or ''for t in tc.findall('.//'+q(A,'t')))
        if raw=='LIBERO+ success ↑':
            para=tc.find('.//'+q(A,'p'));run=para.find(q(A,'r'));assert run is not None
            for child in list(para):
                if child.tag in [q(A,'r'),q(A,'br')]:para.remove(child)
            first=copy.deepcopy(run);first.find(q(A,'t')).text='LIBERO+'
            second=copy.deepcopy(run);second.find(q(A,'t')).text='success ↑'
            para.append(first);E.SubElement(para,q(A,'br'));para.append(second)
    save(parts,path,tree)
    changed={2,6,8,9,17,18}
    for n,path in enumerate(old,1):
        if n not in changed:assert parts[path]==source.read(path)
    for n in [2,9]:
        tree=E.fromstring(parts[old[n-1]])
        assert not tree.findall('.//'+q(P,'pic'))
        assert {x.get('typeface') for x in tree.findall('.//'+q(A,'latin'))}=={'Times New Roman'}
    with zipfile.ZipFile(TMP/'candidate.pptx','w',compression=zipfile.ZIP_DEFLATED)as out:
        for name,data in parts.items():out.writestr(name,data)
    print('18 slides. Corrected native figures and figure copies use Times New Roman. Other slide parts preserved.')
