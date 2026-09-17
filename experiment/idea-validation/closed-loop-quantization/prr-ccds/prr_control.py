"""CCDS small control files and SLURM only; no login-node compute/bulk API."""
from pathlib import Path
import argparse
import re
import shlex
import sys
import paramiko

PROJECT = Path(__file__).resolve().parents[4]
LIMIT = 65536

def connect():
    fields = {}
    for line in (PROJECT / 'credentials.env').read_text(encoding='utf-8-sig').splitlines():
        if line.strip() and not line.lstrip().startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            fields[key.strip()] = value.strip()
    host = fields.get('CCDS_HOST', '')
    if host != '10.96.189.11':
        raise RuntimeError('CCDS_HOST must match the user guide endpoint')
    user = fields.get('CCDS_USERNAME', '').lower()
    if not re.fullmatch(r'[a-z0-9._-]+', user) or not fields.get('CCDS_PASSWORD'):
        raise RuntimeError('Missing or malformed CCDS credential fields')
    client = paramiko.SSHClient()
    client.load_host_keys(str(Path.home() / '.ssh' / 'known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(host, username=user, password=fields['CCDS_PASSWORD'],
                   allow_agent=False, look_for_keys=False, timeout=20,
                   banner_timeout=20, auth_timeout=20)
    return client, user

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['inventory', 'init', 'upload', 'submit', 'status', 'read'])
    parser.add_argument('values', nargs='*')
    args = parser.parse_args()
    client, user = connect()
    try:
        base = '/tc1home/UG/' + user + '/prr_ccds'
        if args.action in ('upload', 'read'):
            with client.open_sftp() as sftp:
                sftp.get_channel().settimeout(40)
                if args.action == 'upload':
                    for value in args.values:
                        path = Path(value).resolve()
                        assert PROJECT in path.parents and path.suffix in ('.py', '.sh', '.json', '.md')
                        assert not re.search('credential|password|private', path.name, re.I)
                        data = path.read_bytes().replace(b'\r\n', b'\n')
                        assert len(data) <= LIMIT
                        remote = base + '/control/' + path.name
                        with sftp.open(remote+'.uploading', 'wb') as stream:
                            stream.write(data)
                        sftp.posix_rename(remote+'.uploading', remote)
                        print('Uploaded control:', path.name, len(data))
                else:
                    relative = args.values[0]
                    assert '..' not in relative.split('/') and not relative.startswith('/')
                    assert relative.endswith(('.json', '.jsonl', '.txt', '.log', '.out', '.err'))
                    remote = base+'/'+relative
                    assert sftp.stat(remote).st_size <= LIMIT, 'Full arrays stay on compute'
                    with sftp.open(remote, 'rb') as stream:
                        data = stream.read(LIMIT+1)
                    assert len(data) <= LIMIT
                    if len(args.values) > 1:
                        local = Path(args.values[1]).resolve()
                        assert PROJECT in local.parents
                        local.parent.mkdir(parents=True, exist_ok=True)
                        local.write_bytes(data)
                    print(data.decode('utf-8', errors='replace'))
            return
        if args.action == 'inventory':
            command = '\n'.join(['hostname', 'pwd', 'module load slurm',
                'sacctmgr -nP show user '+shlex.quote(user)+' withassoc format=user,account,qos',
                'sacctmgr -nP show qos normal format=name,MaxTRESPU,MaxJobsPU,MaxSubmitPU,MaxWall',
                'sinfo -h -o "%P %a %l %D %t %G"',
                'squeue -u '+shlex.quote(user)+' -o "%.16i %.20j %.8T %.10M %R"',
                'ls -l "$HOME"/cem_update_ccds/control/smoke_runner.py "$HOME"/cem_update_ccds/control/screen_runner.py "$HOME"/cem_update_ccds/control/probe_runner.py 2>/dev/null || true'])
        elif args.action == 'init':
            command = 'mkdir -p '+shlex.quote(base+'/control')+' '+shlex.quote(base+'/artifacts')
        elif args.action == 'submit':
            name = args.values[0]
            assert re.fullmatch(r'[a-z][a-z0-9_]*\.sh', name)
            command = 'cd '+shlex.quote(base) + ' && sbatch --parsable '+shlex.quote(base+'/control/'+name)
        else:
            job = args.values[0]
            assert job.isdigit()
            command = 'sacct -nP -j '+job+' --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES,NodeList\nsqueue -h -j '+job+' -o "%i %T %M %R"\nif [ "$(squeue -h -j '+job+' -o %T)" = RUNNING ]; then sstat -nP -j '+job+'.batch --format=JobID,AveCPU,MaxRSS; fi'
        _, stdout, stderr = client.exec_command(command, timeout=45)
        data = stdout.read(LIMIT)
        error = stderr.read(LIMIT)
        if args.action == 'status':
            folder = PROJECT/'experiment/idea-validation/closed-loop-quantization/prr-ccds/artifacts'/args.values[0]
            folder.mkdir(parents=True, exist_ok=True)
            (folder/'slurm_status.txt').write_bytes(data)
        print(data.decode('utf-8', errors='replace'), end='')
        print(error.decode('utf-8', errors='replace'), file=sys.stderr, end='')
        raise SystemExit(stdout.channel.recv_exit_status())
    finally:
        client.close()

if __name__ == '__main__':
    main()
