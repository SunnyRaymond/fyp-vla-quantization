"""Exercise shell control flow with command stubs; never enters an allocation/workload."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import control_native


def bash_binary():
    if os.name == 'nt':
        git = shutil.which('git')
        candidate = Path(git).parent.parent / 'bin/bash.exe' if git else None
        return str(candidate) if candidate and candidate.is_file() else None
    return shutil.which('bash')


class JobShellTests(unittest.TestCase):
    def setUp(self):
        self.bash = bash_binary()
        if not self.bash:
            self.skipTest('Bash unavailable for shell control tests')

    def test_job_script_shell_syntax(self):
        for path in (HERE / 'job_native.sh', HERE / 'perf_bundle_native.pbs'):
            with self.subTest(path=path.name):
                result = subprocess.run([self.bash, '-n', str(path)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_rotation_diagnostic_pair_has_one_writer_before_model_loading(self):
        source = (HERE / 'job_native.sh').read_text()
        start = source.index('  if [[ "$STAGE" == rotation_parity* ]]')
        block = source[start:source.index('\n  (', start)]
        harness = 'set -euo pipefail\nTASK_ROOT=$NATIVE_CONTROL_FIXTURE\nSTAGE=rotation_parity\nPBS_JOBID=diagnostic-test\n'
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            environment = dict(os.environ, NATIVE_CONTROL_FIXTURE=fixture.as_posix())
            first = subprocess.run([self.bash, '-s'], input=harness + block,
                                   env=environment, capture_output=True, text=True,
                                   encoding='utf-8', errors='replace')
            second = subprocess.run([self.bash, '-s'], input=harness + block,
                                    env=environment, capture_output=True, text=True,
                                    encoding='utf-8', errors='replace')
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 75, second.stderr)
            self.assertEqual((fixture / 'control/rotation_parity_v1_claim/owner.txt').read_text().strip(),
                             'diagnostic-test')

    def test_panel_launches_one_runtime_for_the_full_comparison(self):
        source = (HERE / 'job_native.sh').read_text()
        panel = source[source.index('if [[ "$STAGE" == panel ]]'):
                       source.index('if [[ "$STAGE" == capture ]]')]
        self.assertEqual(panel.count('run_python '), 1)
        self.assertIn('"$ARTIFACTS/source/panel_native.py"', panel)
        self.assertNotIn('--require-gpu-uuid', panel)
        self.assertNotIn('canonical_gpu_uuid.txt', panel)
        self.assertIn('aggregate_native.py', panel)

    def test_termination_does_not_write_a_success_marker(self):
        source = (HERE / 'job_native.sh').read_text()
        traps = source[source.index('monitor_pid='):
                       source.index('echo "PBS_JOBID=')]
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            harness = ('set -euo pipefail\nARTIFACTS=$NATIVE_CONTROL_FIXTURE\n'
                       + traps + '\nkill -TERM "$BASHPID"\nsleep 1\n')
            result = subprocess.run([self.bash, '-s'], input=harness,
                                    env=dict(os.environ, NATIVE_CONTROL_FIXTURE=fixture.as_posix()),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 143, result.stderr)
            self.assertEqual((fixture / 'exit_code.txt').read_text().strip(), '143')
            self.assertFalse((fixture / 'PIPELINE_COMPLETE').exists())

    def test_repack_stage_builders_and_probe_order(self):
        source = (HERE / 'job_native.sh').read_text(encoding='utf-8')
        self.assertIn('STAGE=${NATIVE_STAGE:-$stage_arg}', source)
        self.assertIn('diffusion_repack|svd_fused_repack', source)
        self.assertIn('MAX_JOBS=2', source)
        self.assertIn('builder=build_diffusion.py', source)
        self.assertIn('builder=build_svd_fused.py', source)
        probe = source[source.index('if [[ "$STAGE" == probe_all ]]'):
                       source.index('if [[ "$STAGE" == optimized ]]')]
        self.assertIn('probe_all_worker.py', probe)
        self.assertEqual(probe.count('run_python '), 1)

    def test_wide_branch_runs_the_same_six_work_units_despite_one_failure(self):
        source = (HERE / 'job_native.sh').read_text()
        anchor = source.index('  wide_arms=')
        start = source.rfind('if [[ "$STAGE" == gate_wide* ]]', 0, anchor)
        block = source[start:source.index('if [[ "$STAGE" == gate_* ]]', anchor)]
        harness = r'''
set -euo pipefail
STAGE=gate_wide
ARTIFACTS=$NATIVE_CONTROL_FIXTURE
TASK_ROOT=$NATIVE_CONTROL_FIXTURE
OLD_ROOT=/unused
wide_gpu_ids=(test0 test1 test2 test3 test4 test5)
run_python() {
  printf '%s\n' "$3" >> "$ARTIFACTS/attempted.txt"
  [[ "$3" != viditq_w4a4 ]]
}
'''
        for gpu_count in (1, 3, 4, 6):
            with self.subTest(gpu_count=gpu_count), tempfile.TemporaryDirectory() as temporary:
                fixture = Path(temporary)
                fixture_harness = harness.replace(
                    'wide_gpu_ids=(test0 test1 test2 test3 test4 test5)',
                    'wide_gpu_ids=(' + ' '.join(f'test{i}' for i in range(gpu_count)) + ')')
                result = subprocess.run([self.bash, '-s'], input=fixture_harness + block,
                                        env=dict(os.environ, NATIVE_CONTROL_FIXTURE=fixture.as_posix()),
                                        capture_output=True, text=True)
                arms = (fixture / 'attempted.txt').read_text().splitlines()
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertCountEqual(arms, ['spinquant_w4a4', 'quarot_w4a4', 'svdquant_w4a4',
                                           'viditq_w4a4', 'ours_vqwam', 'bf16'])
        self.assertIn('--variant current', block)

    def test_four_loaded_workers_continue_after_one_fails(self):
        source = (HERE / 'job_native.sh').read_text()
        block = source[source.index('if [[ "$STAGE" == gates ]]'):
                       source.index('if [[ "$STAGE" == svd_fused_gate ]]')]
        harness = r'''
set -euo pipefail
STAGE=gates
ARTIFACTS=$NATIVE_CONTROL_FIXTURE
TASK_ROOT=$NATIVE_CONTROL_FIXTURE
OLD_ROOT=/unused
CUDA_VISIBLE_DEVICES=unit-test0,unit-test1,unit-test2,unit-test3
run_python() {
  printf '%s\n' "$3" >> "$ARTIFACTS/attempted.txt"
  [[ "$3" != spinquant_w4a4 ]]
}
'''
        with tempfile.TemporaryDirectory() as temp:
            fixture = Path(temp)
            result = subprocess.run([self.bash, '-s'], input=harness + block,
                                    env=dict(os.environ, NATIVE_CONTROL_FIXTURE=fixture.as_posix()),
                                    capture_output=True, text=True)
            attempted = (fixture / 'attempted.txt').read_text().splitlines()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertCountEqual(attempted, ['spinquant_w4a4', 'quarot_w4a4',
                                          'svdquant_w4a4', 'viditq_w4a4'])
        self.assertEqual(block.count('run_python '), 1)
        self.assertIn('"$ARTIFACTS/source/gates_worker.py"', block)

    def test_perf_bundle_stage_registration_and_pbs_resources(self):
        self.assertEqual(control_native.STAGES['perf_bundle'],
                         ('cpnp25pbundle', 'perf_bundle_native.pbs'))
        pbs = (HERE / 'perf_bundle_native.pbs').read_text(encoding='utf-8')
        self.assertIn('#PBS -N cpnp25pbundle', pbs)
        self.assertIn('#PBS -q normal', pbs)
        self.assertIn('#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1', pbs)
        self.assertIn('#PBS -l walltime=02:01:00', pbs)
        self.assertIn('job_native.sh perf_bundle', pbs)

    def test_perf_bundle_attempts_all_five_arms_after_one_failure(self):
        source = (HERE / 'job_native.sh').read_text(encoding='utf-8')
        start = source.index('if [[ "$STAGE" == perf_bundle ]]')
        block = source[start:source.index('if [[ "$STAGE" == perf_* ]]', start)]
        harness = r'''
set -euo pipefail
STAGE=perf_bundle
ARTIFACTS=$NATIVE_CONTROL_FIXTURE/artifacts
OLD_ROOT=/unused
TASK_ROOT=$NATIVE_CONTROL_FIXTURE
PBS_JOBID=25990000.pbs101
mkdir -p "$ARTIFACTS"
run_python() {
  printf '%s %s\n' "$3" "$5" >> "$ARTIFACTS/attempted.txt"
  [[ "$3" != quarot_w4a4 ]]
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            result = subprocess.run([self.bash, '-s'], input=harness + block,
                                    env=dict(os.environ, NATIVE_CONTROL_FIXTURE=fixture.as_posix()),
                                    capture_output=True, text=True)
            artifacts = fixture / 'artifacts'
            attempts = (artifacts / 'attempted.txt').read_text().splitlines()
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(attempts, [
                'spinquant_w4a4 current', 'quarot_w4a4 current',
                'svdquant_w4a4 current', 'viditq_w4a4 current',
                'ours_vqwam optimized'])
            for arm in ('spinquant_w4a4', 'quarot_w4a4', 'svdquant_w4a4',
                        'viditq_w4a4', 'ours_vqwam'):
                log = (artifacts / f'perf_bundle_{arm}.log').read_text()
                self.assertIn(f'PBS_JOBID=25990000.pbs101 arm={arm}', log)
                self.assertIn(f'PERF_BUNDLE_RESULT arm={arm}', log)

    def test_formal_race_maps_performance_stages_to_existing_claim_stages(self):
        source = (HERE / 'job_native.sh').read_text(encoding='utf-8')
        case_start = source.index('race_stage=\ncase "$STAGE" in')
        case_end = source.index('\nesac', case_start) + len('\nesac')
        race_start = source.index('  if [[ -n "$race_stage" ]]', case_end)
        race_end = source.index('  if [[ "$STAGE" == gate_wide* ]]', race_start)
        block = source[case_start:case_end] + '\n' + source[race_start:race_end]
        expected = {'perf_spin': 'gate_spin', 'perf_qua': 'gate_qua',
                    'perf_svd': 'gate_svd', 'perf_vidit': 'gate_vidit',
                    'perf_ours': 'ours_gate', 'perf_bundle': 'gate_wide'}
        self.assertLess(source.index('Job_Name[[:space:]]*='), race_start)
        self.assertLess(race_start, source.index('source "$OLD_ROOT/environment.sh"'))
        for actual_stage, virtual_stage in expected.items():
            with self.subTest(stage=actual_stage), tempfile.TemporaryDirectory() as temporary:
                fixture = Path(temporary)
                race_root = fixture / 'control' / 'formal_n40_queue_race'
                race_root.mkdir(parents=True)
                (race_root / 'queue_race.json').write_text('{}')
                harness = f'''
set -euo pipefail
STAGE={actual_stage}
TASK_ROOT=$NATIVE_CONTROL_FIXTURE
ARTIFACTS=$NATIVE_CONTROL_FIXTURE/artifacts
CUDA_VISIBLE_DEVICES=unit-test0
expected_name=unused
single_arm=
mkdir -p "$ARTIFACTS/source"
nvidia-smi() {{
  if [[ "$*" == *query-gpu=name* ]]; then printf '%s\\n' 'NVIDIA A100-SXM4-40GB'; fi
}}
python3() {{ printf '%s\\n' "$*" > "$ARTIFACTS/race-call.txt"; }}
'''
                result = subprocess.run([self.bash, '-s'], input=harness + block,
                                        env=dict(os.environ, NATIVE_CONTROL_FIXTURE=fixture.as_posix()),
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                invocation = (fixture / 'artifacts' / 'race-call.txt').read_text()
                self.assertIn(f'--root {race_root.as_posix()}', invocation)
                self.assertIn(f'--stage {virtual_stage}', invocation)


if __name__ == '__main__':
    unittest.main()
