import json
from pathlib import Path
import tempfile
import unittest

import aggregate_native as aggregate
import perf_native
from perf_native import ARMS, _gate_limits


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = json.loads((Path(__file__).parent / 'native_protocol.json').read_text())
        self.protocol.pop('baseline_fidelity_policy', None)
        self.protocol['selection_status'] = 'ready_for_formal_panel'
        self.write('native_protocol.json', self.protocol)
        hardware = {'gpu_uuid': 'GPU-fixture', 'gpu_name': self.protocol['hardware']['gpu'],
                    'total_memory_bytes': 42406903808, 'capability': [8, 0],
                    'torch_version': '2.7.0+cu128', 'cuda_version': '12.8',
                    'hostname': 'node-a', 'pbs_job_id': 'fixture'}
        for scope in aggregate.SCOPES:
            for index, arm in enumerate(ARMS):
                variant = 'optimized' if arm == 'ours_vqwam' else 'current'
                native = arm not in ('bf16', 'ours_vqwam')
                binding = {'arm': arm, 'requested_variant': variant,
                           'native_int4_kernel_used': native}
                if native:
                    binding['extension_build'] = {'pbs_job_id': 'build-fixture'}
                timing = {'count': 40, 'mean': .4 / (index + 1), 'median': .39 / (index + 1),
                          'p95': .42 / (index + 1), 'max': .45 / (index + 1)}
                summary = {'format': self.protocol['protocol'], 'protocol': self.protocol['source_protocol'],
                           'status': 'complete', 'complete': True, 'scope': scope, 'arm': arm,
                           'variant': variant, 'performance_design': '40ctx_once_v3',
                           'hardware': hardware, 'backend_binding': binding,
                           'install_metadata': {'native_int4_kernel_used': native},
                           'sample_hardware_provenance': {'sample_count': 40,
                               'devices': [{'gpu_uuid': 'GPU-fixture', 'hostnames': ['node-a'],
                                            'pbs_job_ids': ['fixture']}]},
                           'coverage': {'queries': 40, 'samples': 40, 'expected_samples': 40,
                                        'blocks': 1, 'repetitions': 1},
                           'training_updates': 0, 'closed_loop_episodes': 0,
                           'denoising_wall_seconds': timing, 'dit_replay_seconds': timing,
                           'peak_allocated_gb_decimal': {'count': 40, 'max': 5.0},
                           'storage': {'packed_storage_bytes': 2**30,
                                       'packed_storage_gib_binary': 1.0},
                           'graph_capture_count': 5, 'graph_pool_contract': 'fixture',
                           'non_dit_offload': {'modules': []}}
                if scope == 'dit_replay':
                    summary['backend_binding'] = {**binding, 'dit_replay_dispatch': 'fixture'}
                self.write(self.summary_path(scope, arm), summary)
                if arm == 'bf16':
                    legacy = json.loads(json.dumps(summary))
                    legacy.pop('performance_design')
                    legacy['coverage'].update(queries=80, samples=720, expected_samples=720,
                                              blocks=3, repetitions=3)
                    legacy['sample_hardware_provenance']['sample_count'] = 720
                    legacy['denoising_wall_seconds'].update(count=720)
                    legacy['dit_replay_seconds'].update(count=720)
                    legacy['peak_allocated_gb_decimal'].update(count=720)
                    self.write(self.legacy_summary_path(scope, arm), legacy)
                if arm != 'bf16':
                    limits = _gate_limits(scope, arm, variant, full_panel=True)
                    queries = [{'case_id': case, 'seed_index': seed, 'status': 'passed',
                                'metrics': {name: limit / 2 for name, limit in limits.items()}}
                               for case in range(3000, 3040) for seed in (0, 1)]
                    self.write(Path('results/native_gate') / scope / variant / arm / 'full80.json',
                               {'protocol': self.protocol['source_protocol'], 'arm': arm,
                                'variant': variant, 'scope': scope, 'phase': 'full80',
                                'status': 'passed', 'completed_queries': 80,
                                'expected_queries': 80, 'queries': queries,
                                'thresholds': limits, 'backend_binding': binding})

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def summary_path(self, scope, arm):
        variant = 'optimized' if arm == 'ours_vqwam' else 'current'
        return Path(perf_native.performance_root(self.root).relative_to(self.root)) / scope / variant / arm / 'summary.json'

    def legacy_summary_path(self, scope, arm):
        variant = 'optimized' if arm == 'ours_vqwam' else 'current'
        return Path('results/native_performance') / scope / variant / arm / 'summary.json'

    def change_summary(self, scope, arm, mutate):
        path = self.summary_path(scope, arm)
        value = json.loads((self.root / path).read_text())
        mutate(value)
        self.write(path, value)

    def gate_path(self, scope, arm, phase='full80'):
        variant = 'optimized' if arm == 'ours_vqwam' else 'current'
        return self.root / 'results/native_gate' / scope / variant / arm / f'{phase}.json'

    def change_gate(self, scope, arm, phase, mutate):
        path = self.gate_path(scope, arm, phase)
        value = json.loads(path.read_text())
        mutate(value)
        self.write(path.relative_to(self.root), value)

    def test_two_scopes_keep_units_speedup_and_all_arms(self):
        report = aggregate.collect(self.root)
        self.assertEqual(len(report['rows']), 12)
        self.assertEqual(report['rows'][0]['measurement_design'], aggregate.LEGACY_DESIGN)
        self.assertEqual(report['rows'][0]['samples'], 720)
        self.assertAlmostEqual(report['rows'][0]['latency_mean_ms'], 400)
        sampler_spin = next(row for row in report['rows']
                            if row['scope'] == 'sampler' and row['arm'] == 'spinquant_w4a4')
        self.assertEqual(sampler_spin['measurement_design'], '40ctx_once_v3')
        self.assertEqual(sampler_spin['samples'], 40)
        sampler_bf16 = report['rows'][0]
        self.assertEqual(sampler_bf16['cohort'], {'queries': 80,
                         'case_ids': list(range(3000, 3040)), 'seed_indices': [0, 1]})
        self.assertEqual(sampler_spin['cohort']['seed_indices'], [0])
        self.assertAlmostEqual(report['rows'][1]['speedup_mean'], 2)
        self.assertEqual(report['rows'][0]['packed_storage_gib_binary'], 1)
        self.assertIn('720 |', aggregate.render(report))
        self.assertIn('| 40 | 200.000 |', aggregate.render(report))
        self.assertIn('| Method | Cohort q | Seeds | Design | N |', aggregate.render(report))
        self.assertIn('independent', aggregate.render_html(report))

    def test_partial_panel_is_not_a_completed_table(self):
        self.change_summary('dit_replay', 'spinquant_w4a4', lambda row: row['coverage'].update(samples=39))
        with self.assertRaisesRegex(ValueError, 'active 40-context summary is incomplete'):
            aggregate.collect(self.root)

    def test_partial_legacy_720_uses_complete_40ctx_and_is_marked_unused(self):
        legacy = self.root / self.legacy_summary_path('sampler', 'bf16')
        active = self.root / self.summary_path('sampler', 'bf16')
        self.assertTrue(legacy.is_file())
        self.assertTrue(active.is_file())
        value = json.loads(legacy.read_text())
        value['coverage']['samples'] = 719
        self.write(self.legacy_summary_path('sampler', 'bf16'), value)
        row = next(row for row in aggregate.collect(self.root)['rows']
                   if row['scope'] == 'sampler' and row['arm'] == 'bf16')
        self.assertEqual(row['measurement_design'], '40ctx_once_v3')
        self.assertEqual(row['samples'], 40)
        self.assertIn('native_performance_n40', row['summary'])
        self.assertEqual(Path(row['unused_incomplete_legacy_summary']),
                         self.legacy_summary_path('sampler', 'bf16'))

    def test_partial_legacy_without_complete_40ctx_is_rejected(self):
        legacy = self.root / self.legacy_summary_path('sampler', 'bf16')
        active = self.root / self.summary_path('sampler', 'bf16')
        value = json.loads(legacy.read_text())
        value['coverage']['samples'] = 719
        self.write(self.legacy_summary_path('sampler', 'bf16'), value)
        active.unlink()
        with self.assertRaisesRegex(ValueError, 'no complete 40-context summary'):
            aggregate.collect(self.root)

    def test_new_40_summary_is_used_when_no_legacy_summary_exists(self):
        (self.root / self.legacy_summary_path('sampler', 'bf16')).unlink()
        row = next(row for row in aggregate.collect(self.root)['rows']
                   if row['scope'] == 'sampler' and row['arm'] == 'bf16')
        self.assertEqual(row['measurement_design'], '40ctx_once_v3')
        self.assertEqual(row['samples'], 40)
        self.assertEqual(row['cohort']['seed_indices'], [0])
        self.assertIn('native_performance_n40', row['summary'])

    def test_same_sku_different_physical_gpu_is_accepted_with_provenance(self):
        def move_device(row):
            row['hardware'].update(gpu_uuid='GPU-other', hostname='node-b', pbs_job_id='job-other')
            row['sample_hardware_provenance']['devices'] = [
                {'gpu_uuid': 'GPU-fixture', 'hostnames': ['node-a'], 'pbs_job_ids': ['fixture']},
                {'gpu_uuid': 'GPU-other', 'hostnames': ['node-b'], 'pbs_job_ids': ['job-other']}]
        self.change_summary('dit_replay', 'ours_vqwam', move_device)
        report = aggregate.collect(self.root)
        self.assertEqual(report['hardware']['measured_device_count'], 2)
        self.assertEqual(report['hardware']['measured_gpu_uuids'], ['GPU-fixture', 'GPU-other'])
        other = next(row for row in report['hardware']['device_provenance']
                     if row['gpu_uuid'] == 'GPU-other')
        self.assertEqual(other['pbs_job_ids'], ['job-other'])
        self.assertIn('2 台', aggregate.render(report))

    def test_incomplete_device_provenance_is_rejected(self):
        self.change_summary('dit_replay', 'spinquant_w4a4', lambda row: row['sample_hardware_provenance']
                            .update(sample_count=39))
        with self.assertRaisesRegex(ValueError, 'active 40-context summary is incomplete'):
            aggregate.collect(self.root)

    def test_different_gpu_compatibility_or_software_is_rejected(self):
        self.change_summary('dit_replay', 'ours_vqwam',
                            lambda row: row['hardware'].update(cuda_version='12.9'))
        with self.assertRaisesRegex(ValueError, 'SKU or software'):
            aggregate.collect(self.root)

    def test_gate_backend_must_match_measured_backend(self):
        self.change_summary('sampler', 'quarot_w4a4',
                            lambda row: row['backend_binding']['extension_build'].update(pbs_job_id='new-build'))
        with self.assertRaisesRegex(ValueError, 'validated backend'):
            aggregate.collect(self.root)

    def test_payload_padding_is_counted_in_both_scopes(self):
        self.change_summary('dit_replay', 'spinquant_w4a4', lambda row: row['storage'].update(
            packed_storage_bytes=2**30+128, packed_storage_gib_binary=1+128/2**30))
        with self.assertRaisesRegex(ValueError, 'payload differs'):
            aggregate.collect(self.root)

    def test_report_only_baselines_aggregate_with_raw_failed_and_missing_fidelity(self):
        self.protocol['baseline_fidelity_policy'] = 'report_only'
        self.write('native_protocol.json', self.protocol)
        spin_limits = _gate_limits('sampler', 'spinquant_w4a4', 'current', full_panel=True)
        failed_query = {'case_id': 3000, 'seed_index': 0, 'status': 'failed',
                        'metrics': {name: limit * 4 for name, limit in spin_limits.items()}}
        for arm in ARMS[1:5]:
            for scope in aggregate.SCOPES:
                variant = 'current'
                limits = _gate_limits(scope, arm, variant, full_panel=True)
                query = {'case_id': 3000, 'seed_index': 0, 'status': 'failed',
                         'metrics': {name: limit * 4 for name, limit in limits.items()}}
                self.change_gate(scope, arm, 'full80', lambda gate, query=query: gate.update(
                    status='failed', completed_queries=1, queries=[query]))
        rep_limits = _gate_limits('sampler', 'spinquant_w4a4', 'current', full_panel=False)
        self.write(Path('results/native_gate/sampler/current/spinquant_w4a4/representative4.json'),
            {'protocol': self.protocol['source_protocol'], 'arm': 'spinquant_w4a4',
             'variant': 'current', 'scope': 'sampler', 'phase': 'representative4',
             'status': 'failed', 'completed_queries': 2, 'expected_queries': 4,
             'queries': [failed_query, {**failed_query, 'case_id': 3009}],
             'thresholds': rep_limits,
             'backend_binding': json.loads(self.gate_path('sampler', 'spinquant_w4a4')
                                             .read_text())['backend_binding']})
        missing = self.gate_path('dit_replay', 'viditq_w4a4')
        missing.unlink()

        report = aggregate.collect(self.root)

        self.assertEqual(len(report['rows']), 12)
        self.assertEqual(report['baseline_fidelity_policy'], 'report_only')
        spin_row = next(row for row in report['rows']
                        if row['scope'] == 'sampler' and row['arm'] == 'spinquant_w4a4')
        evidence = spin_row['fidelity_evidence']['phases']
        self.assertEqual(evidence['representative4']['status'], 'failed')
        self.assertEqual(evidence['representative4']['completed_queries'], 2)
        self.assertEqual(evidence['representative4']['recorded_query_count'], 2)
        self.assertEqual(evidence['representative4']['observed_thresholds'], rep_limits)
        self.assertEqual(evidence['full80']['status'], 'failed')
        self.assertEqual(evidence['full80']['completed_queries'], 1)
        self.assertEqual(evidence['full80']['query_status_counts'], {'failed': 1})
        self.assertEqual(evidence['full80']['observed_thresholds'], spin_limits)
        vidit_row = next(row for row in report['rows']
                         if row['scope'] == 'dit_replay' and row['arm'] == 'viditq_w4a4')
        self.assertEqual(vidit_row['fidelity_evidence']['phases']['full80']['status'], 'missing')
        for arm in ARMS[1:5]:
            baseline_rows = [row for row in report['rows'] if row['arm'] == arm]
            self.assertEqual(len(baseline_rows), 2)
            self.assertTrue(all('fidelity_evidence' in row for row in baseline_rows))
        self.assertNotIn('fidelity_evidence', next(row for row in report['rows']
                            if row['scope'] == 'sampler' and row['arm'] == 'ours_vqwam'))
        markdown = aggregate.render(report)
        html = aggregate.render_html(report)
        self.assertIn('Fidelity evidence (report-only)', markdown)
        self.assertIn('timing samples 不能证明 fidelity 通过或 full80 已完成', markdown)
        self.assertIn('Fidelity evidence (report-only)', html)
        self.assertIn('Raw status', html)

    def test_same_fused_epilogue_names_before_and_after_execution_are_equivalent(self):
        for scope in aggregate.SCOPES:
            self.change_summary(scope, 'spinquant_w4a4', lambda row:
                row['backend_binding'].update(adapter_epilogue='fused_cuda_fp32_group_scaled_add'))
            self.change_gate(scope, 'spinquant_w4a4', 'full80', lambda row:
                row['backend_binding'].update(adapter_epilogue='fused_native_scaled_add'))
        self.assertEqual(len(aggregate.collect(self.root)['rows']), 12)

    def test_strict_policy_still_rejects_baseline_fidelity_failure(self):
        self.change_gate('sampler', 'quarot_w4a4', 'full80',
                         lambda gate: gate.update(status='failed', completed_queries=1))
        with self.assertRaisesRegex(RuntimeError, 'fidelity gates must pass'):
            aggregate.collect(self.root)

    def test_report_only_policy_does_not_relax_ours_1e6_gate(self):
        self.protocol['baseline_fidelity_policy'] = 'report_only'
        self.write('native_protocol.json', self.protocol)
        self.change_gate('sampler', 'ours_vqwam', 'full80',
                         lambda gate: gate.update(status='failed', completed_queries=1))
        with self.assertRaisesRegex(RuntimeError, 'fidelity gates must pass'):
            aggregate.collect(self.root)


if __name__ == '__main__':
    unittest.main()
