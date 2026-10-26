from pathlib import Path
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader
import json

ROOT = Path(__file__).resolve().parent
ROOT.mkdir(parents=True, exist_ok=True)
SOURCES = {
    'gift': ('2609.04193v1', None),
    'mowam': ('2609.20709v1', None),
    'realtimewam': ('2610.10079v1', None),
    'quantwams': ('2607.28405v1', Path(r'D:\Downloads\Final Year Project\papers\wam\008-quantwams\paper-arxiv-v1.pdf')),
    'fastwam': ('2603.16666v2', Path(r'D:\Downloads\Final Year Project\papers\wam\004-fast-wam\paper-arxiv-v2.pdf')),
    'steerquant': ('2609.39056v1', Path(r'D:\Downloads\Final Year Project\papers\wam\014-steerquant\paper-arxiv-v1.pdf')),
}

def acquire(item):
    name, (arxiv, local) = item
    path = local or ROOT / f'{name}-{arxiv}.pdf'
    if not path.exists():
        req = Request(f'https://arxiv.org/pdf/{arxiv}', headers={'User-Agent': 'Mozilla/5.0 academic literature review'})
        with urlopen(req, timeout=60) as response:
            data = response.read(40_000_000)
        if not data.startswith(b'%PDF'):
            raise ValueError(f'Not a PDF for {name}')
        path.write_bytes(data)
    reader = PdfReader(path)
    pages = []
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ''
        pages.append(text)
        if any(term in text.lower() for term in ['real-world task', 'real-world experiment', 'real-world evaluation', 'real-world system', 'fig. 8:', 'fig. 9:', 'figure 7:', 'figure 8:', 'fig. 3:']):
            lines = text.splitlines()
            selected = [line for line in lines if any(x in line.lower() for x in ['real-world', 'fig.', 'figure', 'banana', 'drawer', 'bowls', 'mug', 'camera', 'carrot', 'trial', 'demonstration'])]
            print(name, 'physical page', number, '\n' + '\n'.join(selected)[:1800], flush=True)
    (ROOT / f'{name}-text.txt').write_text('\n\n'.join(f'=== PHYSICAL PAGE {n} ===\n{text}' for n,text in enumerate(pages,1)), encoding='utf-8')
    return {'name': name, 'arxiv': arxiv, 'pdf': str(path), 'pages': len(reader.pages)}

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(acquire, SOURCES.items()))
    (ROOT / 'sources.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False), flush=True)
