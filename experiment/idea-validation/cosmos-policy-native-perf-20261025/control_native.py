"""Small-file PBS control for frozen-checkpoint Cosmos native performance work."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
REMOTE = '/scratch/users/ntu/yguo017/cosmos-policy-native-perf-20261025'
STAGES = {'prepare': ('cpnp25prep', 'prepare_native.pbs'),
          'startup': ('cpnp25start', 'startup_native.pbs'),
          'atom_general_build': ('cpnp25atom', 'atom_general_native.pbs'),
          'aggregate': ('cpnp25agg', 'aggregate_native.pbs'),
          'toolkit': ('cpnp25cuda', 'toolkit_native.pbs'),
          'optimized': ('cpnp25opt', 'optimized_native.pbs'),
          'analyze': ('cpnp25trace', 'analyze_native.pbs'),
          'diffusion': ('cpnp25diff', 'diffusion_native.pbs'),
          'capture': ('cpnp25cap', 'capture_native.pbs'),
          'probe': ('cpnp25probe', 'probe_native.pbs'),
          'probe_all': ('cpnp25probeall', 'probe_all_native.pbs'),
          'gates': ('cpnp25gate', 'gates_native.pbs'),
          'ours_gate': ('cpnp25ogate', 'ours_gate_native.pbs'),
          'gate_spin': ('cpnp25gspin', 'gate_one_native.pbs'),
          'gate_qua': ('cpnp25gqua', 'gate_one_native.pbs'),
          'gate_svd': ('cpnp25gsvd', 'gate_one_native.pbs'),
          'gate_vidit': ('cpnp25gvidit', 'gate_one_native.pbs'),
          'gate_wide': ('cpnp25gwide', 'gate_wide_native.pbs'),
          'gate_wide_g1': ('cpnp25gw1', 'gate_wide_g1_native.pbs'),
          'gate_wide_g2': ('cpnp25gw2', 'gate_wide_g2_native.pbs'),
          'gate_wide_g3': ('cpnp25gw3', 'gate_wide_g3_native.pbs'),
          'perf_bf16': ('cpnp25pbf16', 'panel_arm_native.pbs'),
          'perf_spin': ('cpnp25pspin', 'panel_arm_native.pbs'),
          'perf_qua': ('cpnp25pqua', 'panel_arm_native.pbs'),
          'perf_svd': ('cpnp25psvd', 'panel_arm_native.pbs'),
          'perf_vidit': ('cpnp25pvidit', 'panel_arm_native.pbs'),
          'perf_ours': ('cpnp25pours', 'panel_arm_native.pbs'),
          'perf_bundle': ('cpnp25pbundle', 'perf_bundle_native.pbs'),
          'panel': ('cpnp25panel', 'panel_native.pbs'),
          'build': ('cpnp25build', 'build_native.pbs'),
          'build_fused': ('cpnp25qepi', 'build_fused_native.pbs'),
          'hadamard': ('cpnp25fht', 'hadamard_native.pbs'),
          'svd_fused_build': ('cpnp25svdf', 'svd_fused_build_native.pbs'),
          'svd_fused_gate': ('cpnp25svdg', 'svd_fused_gate_native.pbs'),
          'diffusion_repack': ('cpnp25drepack', 'diffusion_native.pbs'),
          'svd_fused_repack': ('cpnp25sfrepack', 'svd_fused_build_native.pbs'),
          'profile': ('cpnp25prof', 'profile_native.pbs')}
STAGES.update({'rotation_parity': ('cpnp25rdiag', 'rotation_parity_native.pbs'),
               'rotation_parity_long': ('cpnp25rdlong', 'rotation_parity_long_native.pbs')})


def remote_module():
    path = HERE.parent / 'fastwam-libero-plus-pilot/remote.py'
    spec = importlib.util.spec_from_file_location('cosmos_native_remote', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def dependency_handles(stages, current_stage):
    result = {}
    for stage in stages or []:
        if stage == current_stage:
            raise ValueError('A stage cannot depend on itself')
        value = json.loads((HERE / (stage + '_handle.json')).read_text())
        job = value.get('job_id', '')
        if not re.fullmatch(r'\d+\.[\w.-]+', job):
            raise ValueError('Invalid registered dependency handle: ' + stage)
        result[stage] = job
    return result


def terminal_failure(status):
    terminal = re.search(r'^\s*job_state\s*=\s*F\s*$', status, re.M)
    code = re.search(r'^\s*Exit_status\s*=\s*(-?\d+)\s*$', status, re.M)
    return bool(terminal and code and int(code.group(1)) != 0)


def failed_dependency_without_execution(status, parent_statuses):
    return bool(re.search(r'^\s*job_state\s*=\s*F\s*$', status, re.M)
                and not re.search(r'^\s*(?:Exit_status|exec_host|resources_used[.\w]*)\s*=', status, re.M)
                and any(terminal_failure(parent) for parent in parent_statuses))


def afterok_jobs(status):
    value = ''
    for line in status.splitlines():
        if re.match(r'^\s*depend\s*=', line):
            value = line.split('=', 1)[1].strip()
        elif value and line.startswith('\t'):
            value += line.strip()
        elif value:
            break
    result = []
    for clause in value.split(','):
        if clause.startswith('afterok:'):
            for token in clause.split(':')[1:]:
                match = re.fullmatch(r'(\d+\.[\w.-]+)(?:@[\w.-]+)?', token)
                if not match:
                    raise RuntimeError('Invalid PBS afterok dependency')
                result.append(match.group(1))
    return result


def proven_failed_chain(status, query_parent, seen=None):
    if terminal_failure(status):
        return True
    if not re.search(r'^\s*job_state\s*=\s*F\s*$', status, re.M) or re.search(
            r'^\s*(?:Exit_status|exec_host|resources_used[.\w]*)\s*=', status, re.M):
        return False
    seen = set() if seen is None else seen
    if len(seen) >= 16:
        return False
    for parent in afterok_jobs(status):
        if parent in seen:
            continue
        seen.add(parent)
        parent_rc, parent_status = query_parent(parent)
        if parent_rc == 0 and proven_failed_chain(parent_status, query_parent, seen):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['upload', 'submit', 'status'])
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--keep-held', action='store_true')
    parser.add_argument('--after-ok', action='append', choices=STAGES, default=[])
    args = parser.parse_args()
    if args.operation != 'upload' and args.stage is None:
        parser.error('--stage is required')
    remote = remote_module()
    jump = transport = None
    try:
        jump, transport = remote.direct_connect()
        if args.operation == 'upload':
            rc, _ = remote.command(transport, 'mkdir -p ' + shlex.quote(REMOTE))
            if rc:
                raise RuntimeError('Cannot create control directory')
            uploaded = []
            with remote.sftp_client(transport) as sftp:
                for path in sorted(HERE.iterdir()):
                    if path.suffix not in ('.py', '.sh', '.pbs', '.json', '.cu', '.cpp', '.h', '.txt') or not path.is_file():
                        continue
                    if path.name.endswith(('_handle.json', '_intent.json', '_status.json')):
                        continue
                    data = path.read_bytes()
                    if len(data) > 256 * 1024:
                        raise RuntimeError('Control file exceeds small-file limit: ' + path.name)
                    target = REMOTE + '/' + path.name
                    temporary = target + '.upload.' + str(os.getpid())
                    with sftp.open(temporary, 'wb') as stream:
                        stream.write(data)
                    sftp.posix_rename(temporary, target)
                    uploaded.append(path.name)
            print(json.dumps({'uploaded': uploaded, 'remote_root': REMOTE}))
            return
        handle = HERE / (args.stage + '_handle.json')
        if args.operation == 'status':
            value = json.loads(handle.read_text())
            rc, output = remote.command(transport, 'qstat -xf ' + shlex.quote(value['job_id']), timeout=30)
            print(output)
            write_json(HERE / (args.stage + '_status.json'),
                       {'job_id': value['job_id'], 'query_rc': rc, 'observed_utc': stamp(), 'qstat': output})
            return
        if handle.exists():
            previous = json.loads(handle.read_text())
            rc, status = remote.command(transport, 'qstat -xf ' + shlex.quote(previous['job_id']), timeout=30)
            parents = {}
            def query_parent(parent):
                parent_rc, parent_status = remote.command(transport, 'qstat -xf ' + shlex.quote(parent), timeout=30)
                if parent_rc == 0:
                    parents[parent] = parent_status
                return parent_rc, parent_status

            if not args.retry_failed or rc or not proven_failed_chain(status, query_parent):
                raise RuntimeError('Retry requires a verified terminal failed handle')
            archive = HERE / (args.stage + '_failed_' + previous['job_id'].replace('.', '_') + '.json')
            write_json(archive, {**previous, 'terminal_qstat': status,
                                'dependency_terminal_qstats': parents, 'verified_utc': stamp()})
            handle.unlink()
        name, script = STAGES[args.stage]
        dependencies = dependency_handles(args.after_ok, args.stage)
        intent = {'stage': args.stage, 'job_name': name, 'script': script,
                  'after_ok_dependencies': dependencies,
                  'created_utc': stamp(), 'remote_root': REMOTE}
        write_json(HERE / (args.stage + '_intent.json'), intent)
        depend = (' -W ' + shlex.quote('depend=afterok:' + ':'.join(dependencies.values()))
                  if dependencies else '')
        stage_overrides = ''
        if args.stage in ('diffusion_repack', 'svd_fused_repack') or args.stage.startswith(('gate_', 'perf_')):
            stage_overrides = (' -N ' + shlex.quote(name) + ' -v '
                               + shlex.quote('NATIVE_STAGE=' + args.stage))
        command = ('cd ' + shlex.quote(REMOTE) + ' && qsub -h' + stage_overrides
                   + depend + ' ' + shlex.quote(script))
        rc, output = remote.command(transport, command)
        job_id = output.strip()
        if rc or not re.fullmatch(r'\d+\.[\w.-]+', job_id):
            raise RuntimeError('qsub did not return one PBS handle: ' + output)
        receipt = {**intent, 'job_id': job_id, 'registered_utc': stamp()}
        if args.keep_held:
            receipt['initial_user_hold_retained'] = True
        write_json(handle, receipt)
        with remote.sftp_client(transport) as sftp:
            with sftp.open(REMOTE + '/' + handle.name, 'w') as stream:
                stream.write(json.dumps(receipt, indent=2) + '\n')
        if args.keep_held:
            print(json.dumps(receipt))
            return
        rc, output = remote.command(transport, 'qrls ' + shlex.quote(job_id))
        if rc:
            raise RuntimeError('Registered PBS job remains held: ' + output)
        print(json.dumps(receipt))
    finally:
        if transport:
            transport.close()
        if jump:
            jump.close()


if __name__ == '__main__':
    main()
