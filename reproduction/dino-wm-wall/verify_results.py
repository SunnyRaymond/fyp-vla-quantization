"""Independently verify a completed Wall evaluation from recorded states."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
p.add_argument('--expected',type=int,default=50)
p.add_argument('--ffmpeg')
a=p.parse_args()
d=a.directory
summary=json.loads((d/'summary.json').read_text())
cases=json.loads((d/'cases.json').read_text())
assert (d/'SUCCESS').exists()
assert len(cases)==a.expected==summary['n_evals']
assert [c['case_id'] for c in cases]==list(range(a.expected))
data=np.load(d/'trajectories.npz',allow_pickle=False)
assert data['states'].shape[0]==a.expected
assert data['normalized_actions'].shape[0]==a.expected
assert np.isfinite(data['states']).all()
assert np.isfinite(data['normalized_actions']).all()
assert np.isfinite(data['goals']).all()
successes=0
if a.ffmpeg:
    ffmpeg=a.ffmpeg
else:
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
for i,c in enumerate(cases):
    steps=c['executed_env_steps']
    assert c['env_seed']==99*i+1
    assert 0<steps<=summary['max_env_steps']
    assert steps<data['states'].shape[1]
    final=data['states'][i,steps]
    goal=data['goals'][i]
    distance=float(np.linalg.norm(goal[:2]-final[:2]))
    success=distance<4.5
    assert bool(success)==c['success']
    assert np.allclose(final,c['final_state'])
    assert np.allclose(goal,c['goal_state'])
    assert np.isclose(distance,c['goal_distance'])
    assert (d/f'case_{i:02d}.mp4').stat().st_size>1000
    subprocess.run([ffmpeg,'-v','error','-i',str(d/f'case_{i:02d}.mp4'),'-f','null','-'],
                   check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=60)
    successes+=int(success)
assert successes==summary['successes']
assert np.isclose(successes/a.expected,summary['success_rate'])
assert np.isclose(summary['official_metrics']['final_eval/success_rate'],summary['success_rate'])
result={'verified':True,'cases':a.expected,'successes':successes,'success_rate':successes/a.expected,
        'checks':['unique IDs','seed schedule','finite states/actions','action budget','independent distance threshold','aggregate metrics','full video decoding']}
(d/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
