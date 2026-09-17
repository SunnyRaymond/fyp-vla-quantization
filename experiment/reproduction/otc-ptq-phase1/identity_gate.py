"""Check small source/config identities and previously verified checkpoint metadata."""
import hashlib
import json
import os
import socket
from pathlib import Path

assert os.environ.get('PBS_JOBID') and 'login' not in socket.gethostname().lower()
assert socket.gethostname().split('.')[0] == os.environ['OTC_ALLOCATION_HOST']
base = Path('/scratch/users/ntu/yguo017/fastwam-smoke')
artifact = Path(os.environ['ARTIFACTS'])
manifest = json.loads((artifact / 'manifest.json').read_text())
expected = json.loads((artifact / 'source_identity.json').read_text())
for path, signature in expected.items():
    actual = hashlib.sha256((base / 'FastWAM' / path).read_text().encode()).hexdigest()
    assert actual == signature, ('source identity mismatch', path)
for key, name in [('checkpoint', 'libero_optional_idm_2cam224.clean.pt'), ('stats', 'libero_optional_idm_2cam224_dataset_stats.json')]:
    st = (base / 'checkpoints' / name).stat()
    assert st.st_size == manifest[key + '_stat']['size'], key
    assert int(st.st_mtime) == manifest[key + '_stat']['mtime'], key
assert (base / 'checkpoints/libero_optional_idm_2cam224.clean.VERIFIED').is_file()
(artifact / 'identity_check.json').write_text(json.dumps({
    'fastwam_revision': manifest['source_revision'],
    'source_signatures_matched': expected,
    'source_normalization': 'universal newlines, UTF-8; Windows CRLF versus Linux LF ignored',
    'checkpoint_metadata_matched': True,
    'checkpoint_content_hash_recomputed': False,
    'checkpoint_integrity_evidence': 'prior smoke CPU allocation verification reused',
}, indent=2))
print('source/config identities and prior-verified checkpoint metadata match', flush=True)
