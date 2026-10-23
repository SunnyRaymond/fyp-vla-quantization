"""Fetch the user's official download inside an approved CPU allocation."""
from __future__ import annotations
import json
import os
from pathlib import Path
import socket
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import urlopen


def main():
    host = socket.gethostname().split('.')[0]
    nodes = Path(os.environ['PBS_NODEFILE']).read_text().split()
    if not os.environ.get('PBS_JOBID') or 'login' in host or host not in {n.split('.')[0] for n in nodes}:
        raise RuntimeError('A real compute allocation is required')
    root = Path(os.environ['ROOT'])
    request_path = root/'control/tokenizer_download.json'
    target = root/'models/tokenizer.pth'
    if target.is_file() and target.stat().st_size == 507609880:
        print('USER_TOKENIZER_PRESENT bytes=507609880', flush=True)
        return
    if os.environ.get('HF_TOKEN_PATH') and Path(os.environ['HF_TOKEN_PATH']).is_file():
        from huggingface_hub import hf_hub_download
        revisions = json.loads((root/'asset_revisions.json').read_text())
        downloaded = Path(hf_hub_download(
            'nvidia/Cosmos-Predict2-2B-Video2World', 'tokenizer/tokenizer.pth',
            revision=revisions['nvidia/Cosmos-Predict2-2B-Video2World'])).resolve()
        if downloaded.stat().st_size != 507609880:
            raise RuntimeError('Official tokenizer length differs from the user file')
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(downloaded, target)
        receipt = dict(source='official frozen Hugging Face repository using existing cluster login',
                       bytes=target.stat().st_size, destination=str(target),
                       pbs_job_id=os.environ['PBS_JOBID'], host=host)
        (root/'tokenizer_transfer_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        request_path.unlink(missing_ok=True)
        print('USER_TOKENIZER_RECEIVED bytes='+str(receipt['bytes']), flush=True)
        return
    if not request_path.is_file():
        raise RuntimeError('User tokenizer transfer request is missing')
    request = json.loads(request_path.read_text())
    url = request['url'].rstrip('\x00\r\n ')
    if any(ord(character) < 33 for character in url):
        raise RuntimeError('Download metadata contains an unexpected control character')
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or parsed.hostname != 'us.aws.cdn.hf.co':
        raise RuntimeError('Expected the official Hugging Face CDN download')
    if int(request['size']) != 507609880:
        raise RuntimeError('Unexpected user tokenizer length')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix('.pth.partial')
    try:
        with urlopen(url, timeout=90) as response, partial.open('wb') as stream:
            while chunk := response.read(4 * 1024 * 1024):
                stream.write(chunk)
    except (HTTPError, URLError, OSError, HTTPException, ValueError) as error:
        # Signed URLs are deliberately kept out of logs and tracebacks.
        raise RuntimeError('Official tokenizer download failed: '+type(error).__name__+
                           ' status='+str(getattr(error, 'code', 'unavailable'))) from None
    if partial.stat().st_size != int(request['size']):
        raise RuntimeError('Tokenizer transfer length differs from the user file')
    partial.replace(target)
    receipt = dict(source='official CDN URL from user-provided local file download metadata',
                   user_local_path=request['local_path'], bytes=target.stat().st_size,
                   destination=str(target), pbs_job_id=os.environ['PBS_JOBID'], host=host)
    (root/'tokenizer_transfer_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    request_path.unlink()
    print('USER_TOKENIZER_RECEIVED bytes='+str(receipt['bytes']), flush=True)


if __name__ == '__main__':
    main()
