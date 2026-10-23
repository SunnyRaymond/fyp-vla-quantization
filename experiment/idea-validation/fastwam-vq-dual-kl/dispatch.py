"""One-shot local control: release the three GPU arms when C80 coefficients freeze."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def save(**values):
    value={'checked_utc':datetime.now(timezone.utc).isoformat(),**values}
    path=HERE/'dispatch_state.json'
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def command(*args):
    result=subprocess.run([sys.executable,str(HERE/'control.py'),*args],capture_output=True,text=True,timeout=120)
    if result.returncode:
        raise RuntimeError('Control command failed: '+' '.join(args)+'; '+result.stderr[-1500:])
    return json.loads(result.stdout)


def main():
    end=time.monotonic()+5400
    errors=0
    while time.monotonic()<end:
        try:
            status=command('status')
        except Exception as exc:
            errors+=1
            save(phase='query_inconclusive',error=str(exc),consecutive_query_errors=errors)
            time.sleep(60)
            continue
        errors=0
        prep=status.get('units',{}).get('prepare',{})
        q=prep.get('scheduler',{})
        if q.get('job_state') in ('F','X') and q.get('Exit_status') not in (None,'0'):
            raise RuntimeError('Preparation failed; no training jobs submitted')
        ready=status.get('training_coefficients_ready') is True
        save(phase='releasing' if ready else 'waiting_for_C80_coefficients',training_coefficients_ready=ready,
             prepare_job_id=prep.get('job_id'),prepare_phase=prep.get('phase'),prepare_queries=prep.get('queries'))
        if ready:
            submitted={}
            for unit in ('action_kl','latent_kl','dual_kl'):
                submitted[unit]=command('submit','--unit',unit)
                save(phase='releasing',submitted=submitted)
            save(phase='complete',submitted=submitted,training_coefficients_ready=True,
                 submission_policy='one attempt per arm; control handles and intents prevent duplicates; no resubmission')
            return
        time.sleep(60)
    raise TimeoutError('Readiness wait exceeded90minutes; no automatic GPU resubmission')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save(phase='failed',error=type(exc).__name__+': '+str(exc))
        raise
