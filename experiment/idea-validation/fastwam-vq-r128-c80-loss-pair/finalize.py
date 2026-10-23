"""Combine two committed Seq8 receipts inside the approved allocation."""
import argparse
import math
import os
from pathlib import Path
from worker import read, save, validate_protocol, ARMS, PROTOCOL


def validate_arm(s, arm, job):
    expected = dict(protocol=PROTOCOL, arm=arm, pbs_jobid=job, complete=True, status='complete',
                    updates=8, exposures=80, target_linears=614, frozen_rotation=True,
                    final_anchor=True, new_teacher_queries=0)
    if any(s.get(k) != v for k,v in expected.items()):
        raise ValueError('Incomplete or differently bound calibration arm: '+arm)
    for key in ('initial_parity_max_abs','formal_parity_max_abs'):
        v=s.get(key)
        if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1e-6:
            raise ValueError('Calibration parity gate failed: '+key)
    beta=s.get('beta')
    if (type(beta) not in (int,float) or not math.isfinite(beta)
            or (arm=='az' and beta!=0) or (arm=='reader_kl' and beta<=0)):
        raise ValueError('Frozen beta differs')
    return s


def main():
    import sampler
    sampler._require_compute_allocation()
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True)
    args=parser.parse_args()
    p=validate_protocol(read(args.protocol))
    out=Path(p['remote_root'])/'results'
    job=os.environ['PBS_JOBID']
    arms={a:validate_arm(read(out/'arms'/a/'summary.json'),a,job) for a in ARMS}
    for a in ARMS:
        history=read(out/'arms'/a/'training.json')
        if (len(history['history'])!=8 or history['updates']!=8 or history['exposures']!=80
                or sorted(c for h in history['history'] for c in h['cases'])!=list(range(2000,2080))):
            raise ValueError('Full C80 update/exposure gate failed')
        if any(not (out/'arms'/a/name).is_file() for name in ('deployment.pt','anchor.pt','anchor.json')):
            raise ValueError('Missing final deployment or tensor anchor')
    prep=read(out/'reader_preparation/summary.json')
    if (not prep['complete'] or prep['heldout_used'] or prep['calibration_case_ids']!=list(range(2000,2080))
            or prep['beta']!=arms['reader_kl']['beta'] or prep['teacher_action_max_abs']!=0):
        raise ValueError('Initial C80 reader beta/teacher replay gate failed')
    if ([h['cases'] for h in read(out/'arms/az/training.json')['history']]
            !=[h['cases'] for h in read(out/'arms/reader_kl/training.json')['history']]):
        raise ValueError('Paired update order differs')
    save(out/'summary.json',dict(protocol=PROTOCOL,pbs_jobid=job,status='complete',complete=True,
         arms=arms,target_linears=614,new_teacher_queries=0,reader_preparation_cases=80,
         fixed_updates=8,exposures_per_arm=80,selection='fixed final update8',
         scope='frozen R0 and shared VQ0; AZ vs AZ+readerKL; decoded-BF16/A4; closed-loop pending'))


if __name__=='__main__':
    main()
