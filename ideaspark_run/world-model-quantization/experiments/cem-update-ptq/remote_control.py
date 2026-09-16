"""Small control files and PBS operations only; deliberately no bulk-transfer API."""
import argparse
from pathlib import Path
import shlex
import sys

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT/'nscc-access'))
from aspire2a_shell import connect
import paramiko

BASE = '/scratch/users/ntu/yguo017/dino-wm-wall/cem_update_v1'
LIMIT = 65536


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['init', 'upload', 'read', 'status', 'submit'])
    p.add_argument('values', nargs='*')
    args = p.parse_args()
    jump, transport = connect()
    try:
        if args.action in ('upload', 'read'):
            sftp = paramiko.SFTPClient.from_transport(transport)
            sftp.get_channel().settimeout(45)
            try:
                if args.action == 'upload':
                    for value in args.values:
                        local = Path(value).resolve()
                        assert local.parent == Path(__file__).resolve().parent
                        assert local.suffix in ('.py', '.pbs', '.json', '.md')
                        data = local.read_bytes().replace(b'\r\n', b'\n')
                        assert len(data) <= LIMIT, 'Not a small control file'
                        destination = BASE+'/control/'+local.name
                        temporary = destination+'.uploading'
                        with sftp.open(temporary, 'wb') as stream:
                            stream.write(data)
                        sftp.posix_rename(temporary, destination)
                        print('Uploaded small control file:', local.name, len(data))
                else:
                    relative = args.values[0]
                    assert '..' not in relative.split('/') and not relative.startswith('/')
                    import re
                    assert relative.endswith(('.json', '.txt', '.log')) or re.search(r'\.o[0-9]+$', relative)
                    remote = BASE+'/'+relative
                    assert sftp.stat(remote).st_size <= LIMIT, 'Bulk results must stay on compute or use an approved transfer path'
                    with sftp.open(remote, 'rb') as stream:
                        data = stream.read(LIMIT+1)
                    assert len(data) <= LIMIT
                    if len(args.values) == 2:
                        local = Path(args.values[1]).resolve()
                        assert Path(__file__).resolve().parent in local.parents
                        local.parent.mkdir(parents=True, exist_ok=True)
                        local.write_bytes(data)
                    print(data.decode('utf-8'))
            finally:
                sftp.close()
            return
        if args.action == 'init':
            command = 'mkdir -p '+shlex.quote(BASE+'/control')
        elif args.action == 'status':
            import re
            job = args.values[0]
            assert re.fullmatch(r'[0-9]+\.pbs[0-9]+', job)
            command = 'qstat -xf '+shlex.quote(job)
        else:
            script = args.values[0]
            assert script in ('prepare_runtime.pbs', 'gpu_stage.pbs', 'verify_stage.pbs')
            tokens = ['qsub']
            if script == 'gpu_stage.pbs':
                stage = args.values[1]
                assert stage in ('prepare_pools', 'benchmark', 'search')
                tokens += ['-v', 'STAGE='+stage]
                if stage == 'search':
                    seconds = int(args.values[2])
                    assert 300 <= seconds <= 6900
                    tokens[-1] += ',MAX_SECONDS='+str(seconds)
                    requested = seconds + 60
                    tokens += ['-l', 'walltime=%02d:%02d:%02d' % (requested//3600, requested%3600//60, requested%60)]
            elif script == 'verify_stage.pbs':
                import re
                search_job = args.values[1]
                assert re.fullmatch(r'[0-9]+\.pbs[0-9]+', search_job)
                tokens += ['-v', 'SEARCH_JOB='+search_job]
            tokens.append(BASE+'/control/'+script)
            command = 'cd '+shlex.quote(BASE+'/control')+' && '+shlex.join(tokens)
        channel = transport.open_session(timeout=30)
        channel.settimeout(60)
        channel.exec_command(command)
        stdout = channel.makefile('rb').read(LIMIT)
        stderr = channel.makefile_stderr('rb').read(LIMIT)
        if args.action == 'status':
            import re
            folder = Path(__file__).resolve().parent/'artifacts'/args.values[0]
            folder.mkdir(parents=True, exist_ok=True)
            filename = 'qstat_final.txt' if re.search(rb'job_state\s*=\s*F\b', stdout) else 'qstat_snapshot.txt'
            (folder/filename).write_bytes(stdout)
        print(stdout.decode('utf-8'), end='')
        print(stderr.decode('utf-8'), file=sys.stderr, end='')
        raise SystemExit(channel.recv_exit_status())
    finally:
        transport.close()
        jump.close()


if __name__ == '__main__':
    main()
