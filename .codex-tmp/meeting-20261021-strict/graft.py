"""Transplant the two native figures while preserving the user's other slide parts."""
import zipfile,pathlib,posixpath,copy,re,json,xml.etree.ElementTree as E
ROOT=pathlib.Path(r'D:/Downloads/Final Year Project')
TMP=ROOT/'.codex-tmp/meeting-20261021-strict'
P='http://schemas.openxmlformats.org/presentationml/2006/main'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG='http://schemas.openxmlformats.org/package/2006/relationships'
E.register_namespace('p',P);E.register_namespace('a',A);E.register_namespace('r',R)
def slide_paths(z):
    rel=E.fromstring(z.read('ppt/_rels/presentation.xml.rels'))
    targets={r.attrib['Id']:posixpath.normpath(posixpath.join('ppt',r.attrib['Target'])).lstrip('/') for r in rel}
    tree=E.fromstring(z.read('ppt/presentation.xml'))
    return [targets[s.attrib['{'+R+'}id']] for s in tree.find('{'+P+'}sldIdLst')]
def scripts(tree):
    # Math remains ordinary, editable PowerPoint runs, with native subscript baselines.
    pattern=re.compile(r'\^\{([^}]+)\}|\^([A-Za-z0-9])|_\{([^}]+)\}|_([A-Za-z0-9]+(?:,[a-z])?)')
    for paragraph in tree.findall('.//{'+A+'}p'):
        for run in list(paragraph):
            node=run.find('{'+A+'}t')
            if node is None or not node.text or not pattern.search(node.text):continue
            if run.find('{'+A+'}rPr') is None:continue
            raw=node.text;pieces=[];start=0
            for m in pattern.finditer(raw):
                if m.start()>start:pieces.append((raw[start:m.start()],None))
                pieces.append((m.group(1)or m.group(2)or m.group(3)or m.group(4),'sup'if m.group(1)or m.group(2)else'sub'));start=m.end()
            if start<len(raw):pieces.append((raw[start:],None))
            index=list(paragraph).index(run);paragraph.remove(run)
            for value,sub in pieces:
                r=copy.deepcopy(run);r.find('{'+A+'}t').text=value
                props=r.find('{'+A+'}rPr')
                if sub:
                    props.attrib['baseline']='27000'if sub=='sup'else'-23000'
                    if 'sz'in props.attrib:props.attrib['sz']=str(round(int(props.attrib['sz'])*.73))
                paragraph.insert(index,r);index+=1
            if '^'in raw or '_{'in raw:
                parent=next((sp for sp in tree.findall('.//{'+P+'}sp')if paragraph in list(sp.iter())),None)
                if parent is not None:
                    fit=parent.find('.//{'+A+'}normAutofit')
                    if fit is not None:fit.attrib['fontScale']='100000'
with zipfile.ZipFile(TMP/'source-user-edited.pptx')as source,zipfile.ZipFile(TMP/'authored.pptx')as authored:
    parts={i.filename:source.read(i.filename)for i in source.infolist()}
    old=slide_paths(source);new=slide_paths(authored)
    assert len(old)==22 and len(new)==2
    for old_index,new_index in [(2,1),(11,2)]:
        orig_path=old[old_index-1];author_path=new[new_index-1]
        tree=E.fromstring(parts[orig_path]);made=E.fromstring(authored.read(author_path))
        orig_tree=tree.find('{'+P+'}cSld/{'+P+'}spTree');made_tree=made.find('{'+P+'}cSld/{'+P+'}spTree')
        assert not made_tree.findall('{'+P+'}pic'),'No raster overlay is permitted'
        for e in list(orig_tree):orig_tree.remove(e)
        for e in list(made_tree):orig_tree.append(copy.deepcopy(e))
        scripts(tree)
        assert len(tree.findall('.//{'+P+'}sp'))>150
        parts[orig_path]=E.tostring(tree,encoding='utf-8',xml_declaration=True)
        # Preserve original layout/notes relationships. Only replace the actual notes text.
        def note_path(z,slide):
            rp=posixpath.join(posixpath.dirname(slide),'_rels',posixpath.basename(slide)+'.rels')
            for r in E.fromstring(z.read(rp)):
                if r.attrib['Type'].endswith('/notesSlide'):return posixpath.normpath(posixpath.join(posixpath.dirname(slide),r.attrib['Target'])).lstrip('/')
        on=note_path(source,orig_path);an=note_path(authored,author_path)
        if on and an:parts[on]=authored.read(an)
    deleted={3,4,12,13}
    prs=E.fromstring(parts['ppt/presentation.xml']);lst=prs.find('{'+P+'}sldIdLst')
    for index,el in enumerate(list(lst),1):
        if index in deleted:lst.remove(el)
    assert len(lst)==18
    parts['ppt/presentation.xml']=E.tostring(prs,encoding='utf-8',xml_declaration=True)
    mapping={};num=0
    for orig,path in enumerate(old,1):
        if orig in deleted:continue
        num+=1;mapping[orig]=num
        if orig in (2,11):continue
        tree=E.fromstring(parts[path]);changed=False
        for sp in tree.findall('.//{'+P+'}sp'):
            strings=sp.findall('.//{'+A+'}t');raw=''.join(t.text or ''for t in strings)
            off=sp.find('.//{'+A+'}xfrm/{'+A+'}off')
            if raw==str(orig) and off is not None and int(off.attrib['x'])>8500000 and int(off.attrib['y'])>4400000:
                strings[0].text=str(num);changed=True
        if orig in (21,22):
            for t in tree.findall('.//{'+A+'}t'):
                if t.text:
                    replace=t.text.replace('slides 2–4','slide 2').replace('slides 11–13','slide 9')
                    changed|=replace!=t.text;t.text=replace
        if changed:parts[path]=E.tostring(tree,encoding='utf-8',xml_declaration=True)
    # Preserve all user edits, apart from page numbers and the two obsolete reference captions.
    protected=[1,17]
    for i in protected:assert parts[old[i-1]]==source.read(old[i-1])
    with zipfile.ZipFile(TMP/'candidate.pptx','w',compression=zipfile.ZIP_DEFLATED)as out:
        for name,data in parts.items():out.writestr(name,data)
    (TMP/'slide-map.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
    print('18 slides; two native diagram pages; user cover and Thankuuu page preserved byte-for-byte.')

