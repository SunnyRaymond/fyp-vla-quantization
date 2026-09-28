"""Small control-file staging and PBS control; never executes numerical work."""
import argparse
import importlib.util
import json
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IDEAS = ROOT.parent
BASE = '/scratch/users/ntu/yguo017/'
HELPER = IDEAS / 'block-local-action-dynamics' / 'cluster_control.py'
spec = importlib.util.spec_from_file_location('round1_control', HELPER)
access = importlib.util.module_from_spec(spec)
spec.loader.exec_module(access)

TRACKS = {
    'oracle': ('block-local-action-dynamics-oracle', 'run_oracle.pbs', 'JOB.json'),
    'visual_prepare': ('cswm-block-local-pilot', 'prepare_visual.pbs', 'JOB_prepare.json'),
    'visual': ('cswm-block-local-pilot', 'run_visual.pbs', 'JOB.json'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'submit', 'status', 'progress', 'fetch'])
    parser.add_argument('--track', choices=TRACKS, required=True)
    parser.add_argument('--job')
    parser.add_argument('--afterok')
    args = parser.parse_args()
    dirname, pbs, jobfile = TRACKS[args.track]
    local = IDEAS / dirname
    remote = BASE + dirname
    tracker = local / jobfile
    jump, transport = access.connect()
    try:
        if args.action == 'stage':
            manifest = json.loads((local / 'STAGE.json').read_text(encoding='utf-8'))
            files = []
            for item in manifest['files']:
                source = (local / item).resolve()
                if source.suffix not in {'.py', '.json', '.md', '.pbs'}:
                    raise ValueError('Only small source/control files may be staged')
                files.append(source)
            names = [p.name for p in files]
            if len(names) != len(set(names)):
                raise ValueError('Duplicate staged basenames')
            if sum(p.stat().st_size for p in files) > 200_000:
                raise ValueError('Staging exceeds the 200KB small-control-file limit')
            access.command(transport, 'mkdir -p ' + shlex.quote(remote + '/source') + ' ' + shlex.quote(remote + '/runs'))
            sftp = access.paramiko.SFTPClient.from_transport(transport)
            try:
                for source in files:
                    target = remote + '/source/' + source.name
                    with sftp.open(target + '.uploading', 'wb') as stream:
                        stream.write(source.read_bytes().replace(b'\r\n', b'\n'))
                    sftp.posix_rename(target + '.uploading', target)
            finally:
                sftp.close()
            print(f'Staged {len(files)} small control files for {args.track}')
            return
        if args.action == 'submit':
            if tracker.exists():
                raise RuntimeError('Tracked job exists: inspect it; never duplicate a submission')
            command = 'cd ' + shlex.quote(remote + '/source') + ' && qsub '
            if args.afterok:
                if not all(c.isalnum() or c in '._-' for c in args.afterok):
                    raise ValueError('Invalid dependency job id')
                command += '-W ' + shlex.quote('depend=afterok:' + args.afterok) + ' '
            command += shlex.quote(pbs)
            job = access.command(transport, command).strip()
            if not job or not all(c.isalnum() or c in '._-' for c in job):
                raise RuntimeError('Uncertain submission response: inspect PBS before another action')
            tracker.write_text(json.dumps({'job_id': job, 'remote_root': remote,
                'track': args.track, 'afterok': args.afterok}, indent=2) + '\n', encoding='utf-8')
            return
        job = args.job or json.loads(tracker.read_text(encoding='utf-8'))['job_id']
        if not all(c.isalnum() or c in '._-' for c in job):
            raise ValueError('Invalid job id')
        output = remote + '/runs/' + job
        if args.action == 'status':
            access.command(transport, 'qstat -xf ' + shlex.quote(job), compact=True)
        elif args.action == 'progress':
            sftp = access.paramiko.SFTPClient.from_transport(transport)
            try:
                for name in ['progress.json', 'summary_progress.json']:
                    path = output + '/' + name
                    try:
                        if sftp.stat(path).st_size > 200 * 1024:
                            print(f'{name}: summary exceeds bounded retrieval; use job status')
                            continue
                        with sftp.open(path, 'rb') as stream:
                            value = json.loads(stream.read().decode('utf-8'))
                    except FileNotFoundError:
                        continue
                    brief = {key: value[key] for key in ['stage', 'status', 'expected_runs',
                        'completed_count', 'completed_fits', 'current_run', 'current_seed',
                        'current_arm', 'seed', 'arm', 'step', 'elapsed_s', 'expected_fits'] if key in value}
                    if isinstance(value.get('completed_runs'), list):
                        brief['completed_runs'] = len(value['completed_runs'])
                    elif 'completed_runs' in value:
                        brief['completed_runs'] = value['completed_runs']
                    for key in ['completed_stage_a_fits', 'completed_stage_b_fits']:
                        if isinstance(value.get(key), list):
                            brief[key] = len(value[key])
                    print(json.dumps({'file': name, **brief}))
            finally:
                sftp.close()
            statements = []
            for name in ['DONE.json', 'job_status.txt']:
                path = shlex.quote(output + '/' + name)
                statements.append(f'if test -f {path}; then cat {path}; fi')
            path = shlex.quote(output + '/job.log')
            statements.append(f'if test -f {path}; then tail -n 12 {path}; fi')
            access.command(transport, '; '.join(statements))
        else:
            destination = local / 'results' / job
            destination.mkdir(parents=True, exist_ok=True)
            manifest = json.loads((local / 'STAGE.json').read_text(encoding='utf-8'))
            files = manifest.get('prepare_results' if args.track == 'visual_prepare' else 'results', [])
            sftp = access.paramiko.SFTPClient.from_transport(transport)
            try:
                for name in files:
                    if Path(name).name != name:
                        raise ValueError('Result name must be a single basename')
                    path = output + '/' + name
                    if sftp.stat(path).st_size > 200 * 1024:
                        raise ValueError('Result exceeds bounded small-summary retrieval')
                    sftp.get(path, str(destination / name))
            finally:
                sftp.close()
            print(f'Retrieved {len(files)} small result files to {destination}')
    finally:
        transport.close()
        if jump is not None:
            jump.close()


if __name__ == '__main__':
    main()
