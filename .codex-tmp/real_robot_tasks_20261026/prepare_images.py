from pathlib import Path
from urllib.request import urlopen, Request
from html import unescape
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import re, json, subprocess

ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'images'
ASSETS.mkdir(exist_ok=True)

def download(url, name):
    target=ASSETS/name
    if not target.exists():
        with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=60) as response:
            target.write_bytes(response.read(30_000_000))
    with Image.open(target) as image:
        print(name,image.size,flush=True)
    return target

if __name__=='__main__':
    items=[('https://arxiv.org/html/2609.04193v1/setup.png','gift_setup.png'),
           ('https://arxiv.org/html/2609.04193v1/singlearm.png','gift_singlearm.png'),
           ('https://arxiv.org/html/2609.20709v1/images/demo.png','mowam_tasks.png')]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda x:download(*x),items))
    for name, arxiv in [('realtimewam','2610.10079v1'),('quantwams','2607.28405v1')]:
        url=f'https://arxiv.org/html/{arxiv}'
        with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=60) as response:
            html=response.read().decode('utf-8')
        (ROOT/f'{name}.html').write_text(html,encoding='utf-8')
        records=[]
        for match in re.finditer(r'<figure\b.*?</figure>',html,re.S):
            block=match.group()
            caption=re.sub('<[^>]+>',' ',block)
            if any(term in caption.lower() for term in ['real-world','real world']):
                images=re.findall(r'<img\b[^>]*src="([^"]+)"',block)
                records.append({'caption':unescape(re.sub(r'\s+',' ',caption))[:700], 'urls':[urljoin(url,s) for s in images]})
        print(name,json.dumps(records,ensure_ascii=False),flush=True)
        for i,record in enumerate(records):
            for j,image_url in enumerate(record['urls']):
                download(image_url,f'{name}_real_{i}_{j}.png')
