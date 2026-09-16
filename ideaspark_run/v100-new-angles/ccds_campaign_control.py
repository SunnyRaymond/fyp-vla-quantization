"""Small control files and scheduler actions only; no bulk transfer or compute."""
from pathlib import Path
import argparse
import importlib.util
import re
import shlex

PROJECT = Path(__file__).resolve().parents[2]
LIMIT = 65536


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['init', 'upload', 'submit', 'status', 'read'])
    parser.add_argument('values', nargs='*')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('existing_ccds_control', PROJECT / 'nscc-access/ccds_control.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    client, user = helper.connect()
    base = '/tc1home/UG/' + user + '/v100_newangles_ccds'
    try:
        if args.action in ('upload', 'read'):
            with client.open_sftp() as sftp:
                sftp.get_channel().settimeout(40)
                if args.action == 'upload':
                    for value in args.values:
                        path = Path(value).resolve()
                        if PROJECT not in path.parents or path.suffix not in ('.py', '.sh', '.json', '.md'):
                            raise ValueError('Only workspace control files allowed')
                        if re.search('credential|password|private', path.name, re.I):
                            raise ValueError('Credential files forbidden')
                        data = path.read_bytes().replace(b'\r\n', b'\n')
                        if len(data) > LIMIT:
                            raise ValueError('Control file exceeds 64 KiB')
                        remote = base + '/control/' + path.name
                        with sftp.open(remote + '.uploading', 'wb') as stream:
                            stream.write(data)
                        sftp.posix_rename(remote + '.uploading', remote)
                        print('Uploaded control:', path.name, len(data))
                else:
                    relative = args.values[0]
                    if not re.fullmatch(r'artifacts/[0-9]+/[A-Za-z0-9_.-]+\.(json|jsonl|txt|log|out|err)', relative):
                        raise ValueError('Read must name a small job artifact')
                    remote = base + '/' + relative
                    if sftp.stat(remote).st_size > LIMIT:
                        raise ValueError('Full arrays stay on compute storage')
                    with sftp.open(remote, 'rb') as stream:
                        data = stream.read(LIMIT + 1)
                    if len(data) > LIMIT:
                        raise ValueError('Read exceeded limit')
                    if len(args.values) > 1:
                        local = Path(args.values[1]).resolve()
                        if PROJECT not in local.parents:
                            raise ValueError('Output must stay in workspace')
                        local.parent.mkdir(parents=True, exist_ok=True)
                        local.write_bytes(data)
                    print(data.decode('utf-8', errors='replace'))
            return
        if args.action == 'init':
            command = 'mkdir -p ' + shlex.quote(base + '/control') + ' ' + shlex.quote(base + '/artifacts')
        elif args.action == 'submit':
            name = args.values[0]
            if not re.fullmatch(r'[a-z][a-z0-9_]*\.sh', name):
                raise ValueError('Invalid submission filename')
            command = 'module load slurm\ncd ' + shlex.quote(base) + ' && sbatch --parsable ' + shlex.quote(base + '/control/' + name)
        else:
            job = args.values[0]
            if not job.isdigit():
                raise ValueError('Numeric job ID required')
            command = 'module load slurm\nsacct -nP -j ' + job + ' --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES,NodeList\nsqueue -h -j ' + job + ' -o "%i %T %M %R"'
        _, stdout, stderr = client.exec_command(command, timeout=45)
        data, error = stdout.read(LIMIT), stderr.read(LIMIT)
        print(data.decode('utf-8', errors='replace'), end='')
        print(error.decode('utf-8', errors='replace'), end='')
        status = stdout.channel.recv_exit_status()
        if args.action == 'status':
            folder = Path(__file__).resolve().parent / 'scheduler' / args.values[0]
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'slurm_status.txt').write_bytes(data + error)
        raise SystemExit(status)
    finally:
        client.close()


if __name__ == '__main__':
    main()
