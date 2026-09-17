"""Extract small provenance receipts from the completed CPU preparation."""
import hashlib
import json
import os
from pathlib import Path

from allocation_guard import require_allocation
allocation = require_allocation()
top = Path('/tc1home/UG/yguo017/v100_newangles_ccds')
out = top/'artifacts'/os.environ['SLURM_JOB_ID']
path = top/'rounding_persistence_ready/manifest.json'
payload = path.read_bytes()
manifest = json.loads(payload)
brief = {key:value for key,value in manifest.items() if key not in ('source_files','samples')}
brief.update(allocation=allocation,full_manifest_path=str(path),
             full_manifest_sha256=hashlib.sha256(payload).hexdigest(),
             full_manifest_bytes=len(payload),source_file_count=len(manifest['source_files']))
brief['samples'] = [{key:value for key,value in row.items() if key!='source_identities'} for row in manifest['samples']]
encoded = json.dumps(brief,indent=2)
if len(encoded.encode())>=64000:
    raise ValueError('Compact receipts exceed bound')
(out/'manifest_brief.json').write_text(encoded)
print(json.dumps({'full_manifest_sha256':brief['full_manifest_sha256'],'samples':len(brief['samples'])}))
