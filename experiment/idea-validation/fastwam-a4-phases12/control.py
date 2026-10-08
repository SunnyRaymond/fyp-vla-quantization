"""Small-file SSH control with durable PBS submission intents for phase 1/2."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'fastwam-libero-plus-pilot'))
import remote
from array_sources import read_compact_qstat

ROOT = '/scratch/users/ntu/yguo017/fastwam-a4-phases12-20261006'
FILES = ('protocol.json', 'environment.sh', 'prepare.pbs', 'prepare_inputs.py',
         'stage_quantization.py', 'test_stage_quantization.py', 'run.pbs',
         'run_diagnostic.py', 'test_run_diagnostic.py', 'aggregate.pbs',
         'aggregate.py', 'test_aggregate.py')
HANDLE = re.compile(r'^(\d+)(?:\[(\d*)\])?\.([A-Za-z0-9_.-]+)$')

def native_command(_transport, command, timeout=60):
    env=os.environ.copy()
    env.update(NSCC_CREDENTIAL_FILE=str(remote.PROJECT/'credentials.env'),
               SSH_ASKPASS=str(remote.PROJECT/'nscc-access'/'nscc-askpass.cmd'),
               SSH_ASKPASS_REQUIRE='force', DISPLAY='1')
    result=subprocess.run([r'C:\Windows\System32\OpenSSH\ssh.exe','-T',
        '-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2',
        'ASPIRE2A',command],env=env,text=True,capture_output=True,timeout=timeout+30)
    if result.stderr: print(result.stderr.rstrip(),file=sys.stderr)
    return result.returncode,result.stdout

def save(path, value):
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        name=stream.name
    os.replace(name, path)

def remember(phase, receipt):
    path=HERE/'handles.json'
    handles=json.loads(path.read_text()) if path.exists() else {}
    handles[phase]=receipt
    save(path,handles)

def phase_handle(phase):
    path=HERE/'handles.json'
    handles=json.loads(path.read_text()) if path.exists() else {}
    if phase in handles: return handles[phase]['job_id']
    return json.loads((HERE/f'{phase}_handles_a.json').read_text())['job_id']

def submit(transport, phase, attempt):
    configs = {
        'prepare': ('fw12p', 'prepare.pbs', ''),
        'preflight': ('fw12g', 'run.pbs', '-v FW_CASE_ID=0'),
        'remaining': ('fw12a', 'run.pbs', '-J 1-21%10'),
        'aggregate': ('fw12s', 'aggregate.pbs', ''),
    }
    prefix, script, options = configs[phase]
    name=f'{prefix}1006{attempt}'
    path=HERE / f'{phase}_handles_{attempt}.json'
    receipt=json.loads(path.read_text()) if path.exists() else {}
    if receipt.get('job_id'):
        remember(phase,receipt)
        print('CONFIRMED_HANDLE', receipt['job_id']); return
    if phase in ('preflight','remaining'):
        required='prepare' if phase=='preflight' else 'preflight'
        parent=phase_handle(required)
        if not parent or not HANDLE.fullmatch(parent):
            raise RuntimeError(f'No confirmed {required} handle; no submission performed')
        code,output=remote.command(transport,f'qstat -xf {shlex.quote(parent)}',timeout=35)
        state=re.search(r'^\s*job_state\s*=\s*(\w+)',output,re.M)
        exit_code=re.search(r'^\s*Exit_status\s*=\s*(-?\d+)',output,re.M)
        if code or not state or state.group(1) not in ('F','X') or not exit_code or exit_code.group(1)!='0':
            raise RuntimeError(f'{required} is not a verified successful terminal job')
        code,output=remote.command(transport,
            f'test -f {ROOT}/artifacts/{shlex.quote(parent)}/PIPELINE_COMPLETE && '
            f'test "$(cat {ROOT}/artifacts/{shlex.quote(parent)}/exit_code.txt)" = 0 && '
            f'test -s {ROOT}/'+('plan.json' if phase=='preflight' else 'results/case_00/case_summary.json'),timeout=25)
        if code: raise RuntimeError(f'{required} completion artifacts are missing')
    if phase=='aggregate':
        code,output=remote.command(transport,f'test -s {ROOT}/terminal_evidence.json',timeout=25)
        if code: raise RuntimeError('Terminal evidence for the 22 diagnostic cases is missing')
    code, output=remote.command(transport, f'qselect -x -u yguo017 -N {shlex.quote(name)} 2>&1', timeout=35)
    if code not in (0,1) or (code==1 and output.strip()):
        raise RuntimeError('PBS history unavailable; no submission performed')
    ids=output.split()
    if ids:
        parsed=[HANDLE.fullmatch(item) for item in ids]
        if not all(parsed): raise RuntimeError('Unexpected PBS history output')
        bases={(m.group(1),m.group(3)) for m in parsed}
        if len(bases)!=1: raise RuntimeError('Ambiguous PBS submission history')
        base,server=next(iter(bases))
        handle=f'{base}[].{server}' if phase=='remaining' else f'{base}.{server}'
        receipt.update(job_id=handle, status='recovered')
        save(path,receipt); remember(phase,receipt); print('RECOVERED_HANDLE',handle); return
    if receipt:
        raise RuntimeError('Unconfirmed durable intent exists; inspect PBS history before a new attempt')
    receipt={'phase':phase,'attempt':attempt,'job_name':name,'script':script,'options':options,
             'created_utc':datetime.now(timezone.utc).isoformat(),'status':'intent_before_qsub'}
    save(path,receipt)
    code, output=remote.command(transport, f'qsub -q normal -N {shlex.quote(name)} {options} {ROOT}/{script}', timeout=40)
    match=HANDLE.fullmatch(output.strip())
    if code or not match:
        receipt.update(status='unknown_or_rejected',message=output[:2000]); save(path,receipt)
        raise RuntimeError('qsub outcome unconfirmed; recover this attempt before considering another')
    receipt.update(job_id=output.strip(),status='confirmed'); save(path,receipt)
    remember(phase,receipt)
    print('CONFIRMED_HANDLE',receipt['job_id'])

def collect(transport):
    preflight=phase_handle('preflight')
    parent=phase_handle('remaining')
    code,children,parent_state,errors,unexpected,duplicates=read_compact_qstat(transport,parent)
    if code or errors or unexpected or duplicates:
        raise RuntimeError('Incomplete or ambiguous scheduler observation; no evidence published')
    code,output=remote.command(transport,f'qstat -xf {shlex.quote(preflight)}',timeout=35)
    state=re.search(r'^\s*job_state\s*=\s*(\w+)',output,re.M)
    status=re.search(r'^\s*Exit_status\s*=\s*(-?\d+)',output,re.M)
    if code or not state: raise RuntimeError('Preflight scheduler observation failed')
    cases=[{'case_id':0,'job_id':preflight,'state':state.group(1),
            'exit_status':int(status.group(1)) if status else None}]
    for index in range(1,22):
        rows=children.get(index,[])
        if len(rows)!=1: raise RuntimeError(f'Missing/ambiguous array child {index}; re-observe same handle')
        row=rows[0]
        cases.append({'case_id':index,'job_id':row['pbs_jobid'],'state':row['state'],
                      'exit_status':row['exit_status']})
    snapshot={'protocol':'fastwam-a4-phases12-v1',
              'observed_utc':datetime.now(timezone.utc).isoformat(),
              'parent_state':parent_state,'cases':cases}
    save(HERE/'scheduler_snapshot.json',snapshot)
    if any(row['state'] not in ('F','X') or row['exit_status']!=0 for row in cases):
        print('WAIT_OR_FAILURE',json.dumps(cases)); return
    with remote.sftp_client(transport) as sftp:
        for row in cases:
            artifact=f"{ROOT}/artifacts/{row['job_id']}"
            row['marker_path']=artifact+'/PIPELINE_COMPLETE'
            row['exit_code_path']=artifact+'/exit_code.txt'
            sftp.stat(row['marker_path'])
            if sftp.stat(row['exit_code_path']).st_size>32: raise RuntimeError('Invalid exit-code artifact')
            with sftp.file(row['exit_code_path'],'rb') as stream:
                if stream.read().strip()!=b'0': raise RuntimeError('Job exit-code artifact is not zero')
        data=(json.dumps(snapshot,indent=2)+'\n').encode()
        save(HERE/'terminal_evidence.json',snapshot)
        with sftp.file(ROOT+'/terminal_evidence.json','wb') as stream: stream.write(data)
    print('TERMINAL_EVIDENCE_READY cases=22')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('upload','submit','status','pull','collect'))
    parser.add_argument('--phase',choices=('prepare','preflight','remaining','aggregate'))
    parser.add_argument('--attempt',default='a')
    parser.add_argument('--job')
    parser.add_argument('--files',nargs='+')
    parser.add_argument('--native',action='store_true',help='Use existing trusted Windows SSH aliases for submit/status')
    args=parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9]{0,5}',args.attempt): parser.error('Use a short lowercase attempt')
    jump=transport=None
    try:
        if args.native:
            if args.action not in ('submit','status'): parser.error('--native supports submit/status only')
            remote.command=native_command
        else:
            jump,transport=remote.connect()
        if args.action=='upload':
            code,output=remote.command(transport,f'mkdir -p {ROOT}',timeout=25)
            if code: raise RuntimeError(output)
            with remote.sftp_client(transport) as sftp:
                for name in args.files or FILES:
                    if name not in FILES: raise ValueError('Unrecognized control file')
                    path=HERE/name
                    if not path.exists():
                        if args.files: raise FileNotFoundError(path)
                        continue
                    data=path.read_bytes().replace(b'\r\n',b'\n')
                    if len(data)>remote.LIMIT: raise ValueError('Control file exceeds small-transfer limit')
                    with sftp.file(f'{ROOT}/{name}','wb') as stream: stream.write(data)
                    print('UPLOADED',name,len(data))
        elif args.action=='submit':
            if not args.phase: parser.error('--phase is required')
            submit(transport,args.phase,args.attempt)
        elif args.action=='status':
            if not args.job or not HANDLE.fullmatch(args.job): parser.error('Valid --job required')
            code,output=remote.command(transport,f'qstat -xtf {shlex.quote(args.job)}',timeout=40)
            print('RC',code); print(output)
        elif args.action=='collect':
            collect(transport)
        else:
            if not args.files: parser.error('--files is required')
            with remote.sftp_client(transport) as sftp:
                for name in args.files:
                    relative=Path(name)
                    if relative.is_absolute() or '..' in relative.parts: raise ValueError('Invalid pull path')
                    source=f'{ROOT}/{relative.as_posix()}'
                    info=sftp.stat(source)
                    if info.st_size>remote.LIMIT: raise ValueError('Only small summary/control artifacts can be pulled')
                    data=sftp.file(source,'rb').read()
                    target=HERE/relative; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
                    print('PULLED',relative.as_posix(),len(data))
    finally:
        if transport: transport.close()
        if jump: jump.close()

if __name__=='__main__': main()
