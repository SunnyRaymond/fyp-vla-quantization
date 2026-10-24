"""Retain the first started of the two explicitly authorized PBS handles."""
from datetime import datetime
import json
from pathlib import Path
import shlex

import control
import short_control

HERE = Path(__file__).resolve().parent
STATE = HERE/'race_status.json'
JOBS = {
    '25938083.pbs101': ('vpinf0003', 'g1', '04:00:00'),
    '25938321.pbs101': ('vpinf0004', 'gdev', '01:59:00'),
}


def start_time(value):
    if isinstance(value, (int, float)) or str(value).isdigit():
        return float(value)
    for fmt in ('%a %b %d %H:%M:%S %Y', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except (TypeError, ValueError):
            pass
    raise ValueError('Cannot compare the PBS start times: '+repr(value))


def choose_winner(rows):
    started = [key for key,row in rows.items() if row['job_state'] in ('R','E')
               or (row['job_state']=='F' and row.get('Exit_status')==0 and row.get('stime'))]
    if not started:
        return None
    if len(started)==1:
        return started[0]
    # PBS start times have finite resolution; an equal timestamp favors the earlier submitted long job.
    return min(started, key=lambda key:(start_time(rows[key]['stime']),int(key.split('.')[0])))


def query(transport):
    code,out=control.remote.command(transport,'qstat -xf -F json '+' '.join(JOBS),timeout=35)
    if code:
        raise RuntimeError('PBS race query failed; no cancellation attempted: '+out[:500])
    rows=json.loads(out)['Jobs']
    if set(rows)!=set(JOBS):
        raise RuntimeError('PBS race handle set differs')
    for key,row in rows.items():
        name,queue,wall=JOBS[key]
        resource=row['Resource_List']
        if (row['Job_Name']!=name or not row['Job_Owner'].startswith(control.USER+'@')
                or row['queue']!=queue or resource['walltime']!=wall
                or resource.get('ngpus')!=1 or resource.get('ncpus')!=16):
            raise RuntimeError('PBS identity/resource mismatch; no cancellation: '+key)
    return rows


def main():
    jump,transport=control.direct_connect()
    try:
        for module,key in ((control,'25938083.pbs101'),(short_control.control,'25938321.pbs101')):
            handle,_=module._resolve(transport)
            if not handle or handle['job_id']!=key:
                raise RuntimeError('Durable race handle mismatch')
        rows=query(transport)
        previous=json.loads(STATE.read_text()) if STATE.exists() else {}
        winner=previous.get('winner_job_id') or choose_winner(rows)
        value=dict(checked_utc=control.utc_now(),status='waiting',winner_job_id=winner,
            jobs={key:{k:row.get(k) for k in ('Job_Name','job_state','queue','stime','Exit_status')}
                  for key,row in rows.items()})
        if winner:
            if winner not in JOBS:
                raise RuntimeError('Stored race winner differs from the authorized handles')
            loser=next(key for key in JOBS if key!=winner)
            value.update(status='winner_selected',loser_job_id=loser)
            if rows[loser]['job_state']!='F':
                code,out=control.remote.command(transport,'qdel '+shlex.quote(loser),timeout=35)
                value.update(qdel_return_code=code,qdel_response=out[:500])
                rows=query(transport)
                value['jobs']={key:{k:row.get(k) for k in ('Job_Name','job_state','queue','stime','Exit_status')}
                               for key,row in rows.items()}
                if code and rows[loser]['job_state']!='F':
                    value['status']='cancellation_unconfirmed'
            value['loser_terminal_verified']=rows[loser]['job_state']=='F'
            if value['loser_terminal_verified']:
                value['status']='race_resolved'
        tmp=STATE.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        tmp.replace(STATE)
        print(json.dumps(value,indent=2))
    finally:
        transport.close()
        if jump:jump.close()


if __name__=='__main__':
    main()
