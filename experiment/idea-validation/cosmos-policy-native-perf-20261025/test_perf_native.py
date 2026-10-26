from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import perf_native as worker


@contextmanager
def fake_cosmos_data_type():
    module_name = "cosmos_policy._src.predict2.conditioner"
    names = ("cosmos_policy", "cosmos_policy._src", "cosmos_policy._src.predict2", module_name)
    modules = {name: ModuleType(name) for name in names}
    for module in modules.values():
        module.__path__ = []
    for parent, child in zip(names, names[1:]):
        setattr(modules[parent], child.rsplit(".", 1)[-1], modules[child])
    data_type = Enum("DataType", {"IMAGE": "image", "VIDEO": "video", "MIX": "mix"},
                     module=module_name, qualname="DataType")
    modules[module_name].DataType = data_type
    with patch.dict(sys.modules, modules):
        yield data_type


def frozen_protocol():
    return {
        "protocol": "test-cosmos-v1",
        "mode": "libero_direct_policy",
        "arms": list(worker.ARMS),
        "inference": {"denoising_steps": 5},
        "performance": {"case_ids": list(range(3000, 3040)),
                         "seed_indices": [0, 1], "queries": 80,
                         "blocks": 3, "repetitions": 3, "warmup_queries": 3},
    }


class NativePerfContractTests(unittest.TestCase):
    def test_performance_uses_forty_cases_without_reducing_fidelity_or_capture_cohort(self):
        self.assertEqual(worker.performance_query_keys(frozen_protocol()),
                         [(case, 0) for case in range(3000, 3040)])
        self.assertEqual(len(worker.query_keys(frozen_protocol())), 80)

    def test_interrupted_legacy_sample_is_reused_with_original_device_provenance(self):
        hardware = {'gpu_uuid': 'old-device', 'gpu_name': 'A100', 'total_memory_bytes': 1000,
                    'capability': [8, 0], 'torch_version': '2.7', 'cuda_version': '12.8'}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            new_path = worker.sample_path(root, 'sampler', 'optimized', 'ours_vqwam', 3000, 0, 0, 0)
            legacy = root / 'results/native_performance/sampler/optimized/ours_vqwam/samples' / new_path.name
            expected = worker.sample_identity(protocol='frozen', hardware=hardware, binding={'kernel': 'same'},
                scope='sampler', variant='optimized', arm='ours_vqwam', case_id=3000, seed_index=0,
                block=0, repetition=0, warmup_count=3)
            old = {key: value for key, value in expected.items() if key != 'performance_design'}
            old.update(hostname='original-node', pbs_job_id='original-job', denoising_seconds=.01)
            legacy.parent.mkdir(parents=True)
            legacy.write_text(json.dumps(old))
            fresh_expected = {**expected, 'hardware': {**hardware, 'gpu_uuid': 'new-device'}}
            reused = worker.reusable_performance_sample(new_path, fresh_expected)
            self.assertEqual(reused['performance_design'], '40ctx_once_v3')
            self.assertEqual(reused['hardware']['gpu_uuid'], 'old-device')
            self.assertEqual(reused['pbs_job_id'], 'original-job')
            self.assertEqual(json.loads(legacy.read_text()), old)
            self.assertEqual(reused['timed_sample_reused_from'], str(legacy))
            with self.assertRaisesRegex(RuntimeError, 'backend_binding'):
                worker.reusable_performance_sample(new_path, {**fresh_expected, 'backend_binding': {'kernel': 'changed'}})

    def test_sampler_times_each_frozen_input_once_and_preserves_old_namespace(self):
        calls, saved = [], []
        benchmark = ModuleType('benchmark')
        def run_one(_common, _runtime, context, **_kwargs):
            calls.append(context)
            return {'denoising_seconds': .01, 'complete_endpoint_query_seconds': .02}
        benchmark.run_one = run_one
        torch = SimpleNamespace(cuda=SimpleNamespace(
            reset_peak_memory_stats=lambda device: None, max_memory_allocated=lambda device: 1000))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / 'results/native_performance/sentinel.json'
            old.parent.mkdir(parents=True)
            old.write_text('old720')
            with patch.dict(sys.modules, {'benchmark': benchmark}), \
                    patch.object(worker, '_load_context', side_effect=lambda _c, _r, case, seed: (case, seed)), \
                    patch.object(worker, 'reusable_sample', return_value=None), \
                    patch.object(worker, '_atomic_json', side_effect=lambda path, row: saved.append((path, row))), \
                    patch.object(worker, '_bounded_summary', return_value=100), \
                    patch.object(worker, 'sample_hardware_provenance', return_value={'sample_count': 40}):
                result = worker._run_sampler_benchmark(None, None, frozen_protocol(), root, root,
                    root, 'bf16', 'current', torch, 'fixture', {}, {}, {},
                    {'packed_storage_bytes': 1, 'packed_storage_gib_binary': 1 / 2**30}, float('inf'))
            self.assertEqual(old.read_text(), 'old720')
        self.assertEqual(len(calls), 43)  # Three untimed warmups plus forty unique timings.
        self.assertEqual(len(saved), 40)
        self.assertEqual({(row['case_id'], row['seed_index']) for _, row in saved},
                         {(case, 0) for case in range(3000, 3040)})
        self.assertTrue(all(row['block'] == row['repetition'] == 0 for _, row in saved))
        self.assertTrue(all('native_performance_n40' in path.parts for path, _ in saved))
        self.assertEqual(result['coverage']['samples'], 40)
        self.assertEqual(result['coverage']['blocks'], 1)
        self.assertEqual(result['coverage']['repetitions'], 1)
        self.assertEqual(result['performance_design'], '40ctx_once_v3')

    def test_report_only_fidelity_applies_to_selected_native_baselines_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = {'baseline_fidelity_policy': 'report_only',
                        'selected_variants': {arm: 'current' for arm in worker.ARMS[1:5]}}
            (root / 'native_protocol.json').write_text(json.dumps(protocol))
            binding = {'native_int4_kernel_used': True,
                       'extension_build': {'pbs_job_id': 'verified.pbs101', 'source_commit': 'pinned'}}
            for arm in worker.ARMS[1:5]:
                self.assertTrue(worker.baseline_fidelity_is_report_only(protocol, arm))
                worker._require_gate(root, arm, 'current', 'sampler')
                worker.assert_native_gate_binding(root, arm, 'current', binding)
                with self.assertRaisesRegex(RuntimeError, 'proven native extension'):
                    worker.assert_native_gate_binding(root, arm, 'current', {'native_int4_kernel_used': False})
            self.assertFalse(worker.baseline_fidelity_is_report_only(protocol, 'ours_vqwam'))
            with self.assertRaisesRegex(RuntimeError, 'Both DiT and endpoint full80'):
                worker._require_gate(root, 'ours_vqwam', 'optimized', 'sampler')
            with self.assertRaisesRegex(RuntimeError, 'Unselected'):
                worker._require_gate(root, 'spinquant_w4a4', 'native_grouped', 'sampler')
            protocol['baseline_fidelity_policy'] = 'strict'
            (root / 'native_protocol.json').write_text(json.dumps(protocol))
            with self.assertRaisesRegex(RuntimeError, 'Both DiT and endpoint full80'):
                worker._require_gate(root, 'spinquant_w4a4', 'current', 'sampler')

    def test_load_keeps_weights_only_and_scopes_the_exact_version_class(self):
        class TorchVersion(str):
            pass
        active = []
        @contextmanager
        def safe_globals(values):
            self.assertEqual(values, [TorchVersion])
            active.append(True)
            try:
                yield
            finally:
                active.clear()
        def load(path, **kwargs):
            self.assertEqual(active, [True])
            self.assertEqual(path, Path('existing_capture.pt'))
            self.assertEqual(kwargs, {'map_location': 'cpu', 'weights_only': True})
            return {'hardware': {'torch_version': TorchVersion('2.7.0+cu128')}}
        torch = SimpleNamespace(serialization=SimpleNamespace(safe_globals=safe_globals), load=load)
        with patch.object(worker, 'import_module', return_value=SimpleNamespace(TorchVersion=TorchVersion)) as imported:
            result = worker._load_torch(Path('existing_capture.pt'), torch)
        imported.assert_called_once_with('torch.torch_version')
        self.assertEqual(result['hardware']['torch_version'], '2.7.0+cu128')
        self.assertEqual(active, [])
        tensorless = SimpleNamespace(is_tensor=lambda value: False)
        self.assertIs(type(worker._cpu_tree(TorchVersion('2.7'), tensorless)), str)

    def test_same_gpu_uuid_accepts_torch_and_nvidia_smi_formats(self):
        class TorchVersion(str):
            pass
        uuid = "99f12394-fe25-fadc-95dc-73336329a6d6"
        props = SimpleNamespace(uuid=uuid, name="NVIDIA A100-SXM4-40GB",
                                total_memory=42406903808, major=8, minor=0)
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda device: props),
                                __version__=TorchVersion("2.7.0+cu128"), version=SimpleNamespace(cuda="12.8"))
        device = SimpleNamespace(type="cuda")
        receipt = worker._gpu_receipt(torch, device, "GPU-" + uuid)
        self.assertEqual(receipt["gpu_uuid"], uuid)
        self.assertIs(type(receipt['torch_version']), str)
        props.uuid = "GPU-" + uuid
        self.assertEqual(worker._gpu_receipt(torch, device, uuid)["gpu_uuid"], props.uuid)
        with self.assertRaisesRegex(RuntimeError, "GPU UUID differs"):
            worker._gpu_receipt(torch, device, "GPU-aaaaaaaa-1111-2222-3333-444444444444")

    def test_formal_hardware_contract_allows_uuid_change_but_fixes_sku_and_software(self):
        contract = {"hardware": {"gpu": "NVIDIA A100-SXM4-40GB", "capability": [8, 0],
                                 "torch": "2.7.0+cu128", "cuda_runtime": "12.8"}}
        measured = {"gpu_uuid": "GPU-other", "gpu_name": "NVIDIA A100-SXM4-40GB",
                    "total_memory_bytes": 42406903808, "capability": [8, 0],
                    "torch_version": "2.7.0+cu128", "cuda_version": "12.8"}
        worker.validate_hardware_compatibility(contract, measured)
        with self.assertRaisesRegex(RuntimeError, "compatibility contract"):
            worker.validate_hardware_compatibility(contract, {**measured, "cuda_version": "12.9"})

    def test_measurement_rejects_gate_from_older_backend_build(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for scope in ('sampler', 'dit_replay'):
                path = root / 'results/native_gate' / scope / 'current/svdquant_w4a4/full80.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'backend_binding': {'producer': 'build1'}}))
            worker.assert_native_gate_binding(root, 'svdquant_w4a4', 'current',
                                              {'producer': 'build1', 'dit_replay_dispatch': 'generic'})
            with self.assertRaisesRegex(RuntimeError, 'different native backend'):
                worker.assert_native_gate_binding(root, 'svdquant_w4a4', 'current', {'producer': 'build2'})

    def test_diffusion_installer_receives_native_build_root_not_snapshot_dir(self):
        install = unittest.mock.Mock(return_value=({'native_int4_kernel_used': True}, None))
        with patch.object(worker, '_install_reference'), patch.object(worker, '_load_method_adapter',
                return_value=SimpleNamespace(install_native=install)) as load_adapter:
            worker._install_native('runtime', {}, Path('/old'), Path('/new/artifacts/job/work'),
                                   'svdquant_w4a4', 'native_fused', {}, native_root=Path('/new'))
        load_adapter.assert_called_once_with('native_svd_fused')
        install.assert_called_once_with('runtime', {}, Path('/old'), 'svdquant_w4a4',
                                        artifact_dir=Path('/new'))

    def test_native_fused_variant_is_only_for_svdquant(self):
        worker.validate_variant('svdquant_w4a4', 'native_fused')
        worker.validate_variant('ours_vqwam', 'optimized')
        with self.assertRaises(ValueError):
            worker.validate_variant('bf16', 'native_fused')
        worker.validate_variant('quarot_w4a4', 'native_grouped')
        with self.assertRaises(ValueError):
            worker.validate_variant('svdquant_w4a4', 'native_grouped')

    def test_grouped_rotation_binding_uses_public_nunchaku_producer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'results/preparation/native_diffusion_build.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'status': 'complete', 'pbs_job_id': 'nunchaku-core-build',
                'arms': {'svdquant_w4a4': {'module': '/build/nunchaku_svdq_native.so',
                                          'source_commit': 'nunchaku-source'}}}))
            binding = worker._binding({'native_int4_kernel_used': True,
                'native_backend_module': 'nunchaku_svdq_native'}, 'quarot_w4a4', 'native_grouped', root)
            self.assertEqual(binding['extension_build']['pbs_job_id'], 'nunchaku-core-build')

    def test_general_atom_binding_uses_its_own_build_producer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'results/preparation/native_viditq_general_build.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'status': 'complete', 'pbs_job_id': 'atom-general-build',
                'module': '/build/viditq_atom_per_channel_native.so', 'source_commit': 'viditq-source'}))
            binding = worker._binding({'native_int4_kernel_used': True,
                'native_extension': '/build/viditq_atom_per_channel_native.so'},
                'viditq_w4a4', 'current', root)
            self.assertEqual(binding['extension_build']['pbs_job_id'], 'atom-general-build')

    def test_motor_gate_excludes_gripper_from_normalization(self):
        class Action:
            def __init__(self, name):
                self.name = name
            def __getitem__(self, _key):
                return self.name + '_motor'
        def compare(reference, _candidate, _torch):
            return {'nrmse': .02 if reference == 'reference_motor' else .001,
                    'max_abs': .001}
        with patch.object(worker, 'compare_tensor_trees', side_effect=compare):
            metrics = worker._gate_query_metrics('sampler',
                {'action': Action('reference'), 'future': {'f': 'f'}},
                {'action': Action('candidate'), 'future': {'f': 'f'}}, None)
        self.assertEqual(metrics['normalized_motor_rmse'], .02)
        self.assertFalse(worker._gate_pass(metrics, {'normalized_motor_rmse': .01}))

    def test_artifact_snapshot_uses_explicit_old_root_for_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('common.py', 'runtime.py'):
                (root / name).write_text('# fixture\n')
            with patch.object(sys, 'path', list(sys.path)), patch.object(sys, 'dont_write_bytecode'):
                self.assertEqual(worker._prepare_code_imports(root, None), root.resolve())

    def test_protocol_keeps_full_frozen_panel(self):
        binding = worker.validate_protocol(frozen_protocol(), "spinquant_w4a4")
        self.assertEqual(binding["samples_per_arm"], 40)
        protocol = frozen_protocol()
        protocol["performance"]["case_ids"] = list(range(3000, 3039))
        with self.assertRaises(ValueError):
            worker.validate_protocol(protocol, "spinquant_w4a4")

    def test_six_arm_order_is_frozen(self):
        protocol = frozen_protocol()
        protocol["arms"].reverse()
        with self.assertRaises(ValueError):
            worker.validate_protocol(protocol, "bf16")

    def test_sample_path_separates_scope_variant_and_arm(self):
        root = Path("/tmp/new-root")
        path = worker.sample_path(root, "dit_replay", "optimized", "ours_vqwam",
                                  3000, 0, 2, 1)
        self.assertEqual(path.parts[-5:],
            ("dit_replay", "optimized", "ours_vqwam", "samples", "3000_0_b2_r1.json"))
        self.assertIn("dit_replay", path.parts)

    def test_gate_resume_reads_its_separate_hardware_receipt(self):
        hardware = {"gpu_uuid": "GPU-one", "gpu_name": "A100",
                    "total_memory_bytes": 80, "capability": [8, 0],
                    "torch_version": "2.7", "cuda_version": "12.8",
                    "hostname": "node-a", "pbs_job_id": "job-a"}
        saved = {"hardware": worker.compatible_gpu_key(hardware),
                 "hardware_receipt": hardware, "status": "passed"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'gate.json'
            path.write_text(json.dumps(saved))
            expected = {"hardware": worker.compatible_gpu_key(dict(hardware, gpu_uuid="GPU-two"))}
            self.assertEqual(worker.reusable_sample(path, expected), saved)

    def test_resume_accepts_same_gpu_sku_on_another_uuid_and_keeps_receipt(self):
        hardware = {"gpu_uuid": "GPU-test", "gpu_name": "A100",
                    "total_memory_bytes": 80, "capability": [8, 0],
                    "torch_version": "2.7", "cuda_version": "12.8",
                    "hostname": "node-a", "pbs_job_id": "job-a"}
        first = worker.sample_identity(protocol="p", hardware=hardware, binding={"backend": "x"},
            scope="sampler", variant="current", arm="spinquant_w4a4", case_id=3000,
            seed_index=0, block=0, repetition=0, warmup_count=3)
        hardware2 = dict(hardware, hostname="node-b", pbs_job_id="job-b")
        second = worker.sample_identity(protocol="p", hardware=hardware2, binding={"backend": "x"},
            scope="sampler", variant="current", arm="spinquant_w4a4", case_id=3000,
            seed_index=0, block=0, repetition=0, warmup_count=3)
        self.assertEqual(first, second)
        hardware2["gpu_uuid"] = "GPU-other"
        changed_device = worker.sample_identity(protocol="p", hardware=hardware2, binding={"backend": "x"},
            scope="sampler", variant="current", arm="spinquant_w4a4", case_id=3000,
            seed_index=0, block=0, repetition=0, warmup_count=3)
        self.assertNotEqual(first, changed_device)
        self.assertEqual(worker.compatible_gpu_key(hardware),
                         worker.compatible_gpu_key(hardware2))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.json"
            saved = {**first, "hostname": "node-a", "pbs_job_id": "job-a",
                     "denoising_seconds": 1.0}
            path.write_text(json.dumps(saved), encoding="utf-8")
            reused = worker.reusable_sample(path, changed_device)
            self.assertEqual(reused["hardware"]["gpu_uuid"], "GPU-test")
            self.assertEqual((reused["hostname"], reused["pbs_job_id"]), ("node-a", "job-a"))

    def test_sample_hardware_provenance_reports_all_actual_devices(self):
        records = [
            {"hardware": {"gpu_uuid": "GPU-a"}, "hostname": "node-a", "pbs_job_id": "job-a"},
            {"hardware": {"gpu_uuid": "GPU-b"}, "hostname": "node-b", "pbs_job_id": "job-b"},
            {"hardware": {"gpu_uuid": "GPU-a"}, "hostname": "node-a", "pbs_job_id": "job-c"},
        ]
        receipt = worker.sample_hardware_provenance(records, {})
        self.assertEqual(receipt["sample_count"], 3)
        self.assertEqual([row["gpu_uuid"] for row in receipt["devices"]], ["GPU-a", "GPU-b"])
        self.assertEqual(receipt["devices"][0]["pbs_job_ids"], ["job-a", "job-c"])

    def test_resume_rejects_changed_backend_and_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.json"
            path.write_text(json.dumps({"protocol": "p", "backend_binding": {"revision": "a"}}),
                            encoding="utf-8")
            self.assertIsNotNone(worker.reusable_sample(path, {
                "protocol": "p", "backend_binding": {"revision": "a"}}))
            with self.assertRaises(RuntimeError):
                worker.reusable_sample(path, {"protocol": "p", "backend_binding": {"revision": "b"}})

    def test_fixed_gate_thresholds_include_strict_optimized_endpoint_gate(self):
        self.assertEqual(worker._gate_limits("dit_replay", "spinquant_w4a4", "current",
                                             full_panel=True), {"dit_forward_nrmse": 0.01})
        self.assertEqual(worker._gate_limits("dit_replay", "spinquant_w4a4", "current",
                                             full_panel=False),
                         {"dit_forward_nrmse": 0.01, "linear_operator_nrmse": 0.01})
        self.assertEqual(worker._gate_limits("sampler", "ours_vqwam", "optimized",
                                             full_panel=True),
                         {"action_max_abs": 1e-6, "future_max_abs": 1e-6})
        self.assertEqual(worker._gate_limits("sampler", "ours_vqwam", "optimized",
                                             full_panel=False),
                         {"normalized_motor_rmse": 0.01, "future_nrmse": 0.01})
        self.assertEqual(worker._gate_limits("dit_replay", "ours_vqwam", "optimized",
                                             full_panel=True), {"dit_forward_max_abs": 1e-6})

    def test_ours_optimized_strict_gate_uses_max_abs_not_nrmse(self):
        endpoint_limits = worker._gate_limits("sampler", "ours_vqwam", "optimized",
                                              full_panel=True)
        self.assertTrue(worker._gate_pass({"normalized_motor_rmse": 0.5,
                                           "future_nrmse": 0.7,
                                           "action_max_abs": 9e-7,
                                           "future_max_abs": 9e-7}, endpoint_limits))
        self.assertFalse(worker._gate_pass({"normalized_motor_rmse": 0.0,
                                            "future_nrmse": 0.0,
                                            "action_max_abs": 1.01e-6,
                                            "future_max_abs": 0.0}, endpoint_limits))
        dit_limits = worker._gate_limits("dit_replay", "ours_vqwam", "optimized",
                                         full_panel=True)
        self.assertTrue(worker._gate_pass({"dit_forward_nrmse": 0.5,
                                           "dit_forward_max_abs": 9e-7}, dit_limits))
        self.assertFalse(worker._gate_pass({"dit_forward_nrmse": 0.0,
                                            "dit_forward_max_abs": 1.01e-6}, dit_limits))

    def test_performance_requires_full80_max_abs_gate_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for scope in ("sampler", "dit_replay"):
                path = root / "results" / "native_gate" / scope / "optimized" / "ours_vqwam" / "full80.json"
                path.parent.mkdir(parents=True)
                thresholds = (worker._gate_limits(scope, "ours_vqwam", "optimized", full_panel=True))
                metric_names = ("action_max_abs", "future_max_abs") if scope == "sampler" else (
                    "dit_forward_max_abs",)
                queries = [{"case_id": case_id, "seed_index": seed,
                            "metrics": {name: 9e-7 for name in metric_names}, "status": "passed"}
                           for case_id in range(3000, 3040) for seed in (0, 1)]
                path.write_text(json.dumps({"status": "passed", "completed_queries": 80,
                                            "thresholds": thresholds, "queries": queries}),
                                encoding="utf-8")
            worker._require_gate(root, "ours_vqwam", "optimized", "sampler")
            dit_path = root / "results" / "native_gate" / "dit_replay" / "optimized" / "ours_vqwam" / "full80.json"
            dit_summary = json.loads(dit_path.read_text(encoding="utf-8"))
            dit_summary["thresholds"].pop("dit_forward_max_abs")
            dit_path.write_text(json.dumps(dit_summary), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "DiT output max-abs gate"):
                worker._require_gate(root, "ours_vqwam", "optimized", "sampler")
            dit_summary["thresholds"]["dit_forward_max_abs"] = 1e-6
            dit_summary["queries"][0]["metrics"]["dit_forward_max_abs"] = 1.01e-6
            dit_path.write_text(json.dumps(dit_summary), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "query .* failed dit_forward_max_abs"):
                worker._require_gate(root, "ours_vqwam", "optimized", "sampler")

    def test_endpoint_result_uses_future_image_slots_not_future_alias(self):
        action = object()
        future_image_slots = {"future_wrist": object(), "future_image": object()}
        result = SimpleNamespace(action=action, future_image_slots=future_image_slots)
        self.assertFalse(hasattr(result, "future"))
        trees = worker._result_trees(result)
        self.assertIs(trees["action"], action)
        self.assertIs(trees["future"], future_image_slots)

    def test_capture_replay_rejects_opaque_values_without_torch(self):
        class TensorlessTorch:
            @staticmethod
            def is_tensor(_value):
                return False

        self.assertEqual(worker._cpu_tree({"x": [1, None]}, TensorlessTorch), {"x": [1, None]})
        with self.assertRaisesRegex(RuntimeError, "Opaque/non-tensor"):
            worker._cpu_tree(object(), TensorlessTorch)

    def test_cosmos_str_enum_is_saved_as_primitive_marker_and_restored_by_identity(self):
        class TensorlessTorch:
            @staticmethod
            def is_tensor(_value):
                return False

        with fake_cosmos_data_type() as data_type:
            captured = worker._cpu_tree({"data_type": data_type.VIDEO}, TensorlessTorch)
            marker = captured["data_type"]
            self.assertEqual(marker, {worker.COSMOS_ENUM_MARKER: {
                "module": "cosmos_policy._src.predict2.conditioner",
                "qualname": "DataType", "member": "VIDEO"}})
            self.assertTrue(all(type(value) is str for value in marker[worker.COSMOS_ENUM_MARKER].values()))

            restored = worker._device_tree(captured, "cuda:0", TensorlessTorch)
            self.assertIs(restored["data_type"], data_type.VIDEO)
            self.assertEqual(worker._tree_signature({"data_type": data_type.VIDEO}, TensorlessTorch),
                             worker._tree_signature(captured, TensorlessTorch))
            worker._copy_tree_({"data_type": data_type.VIDEO}, captured, TensorlessTorch)
            with self.assertRaisesRegex(RuntimeError, "Cosmos Enum differs"):
                worker._copy_tree_({"data_type": data_type.IMAGE}, captured, TensorlessTorch)

    def test_capture_rejects_non_allowlisted_string_enum(self):
        class OtherDataType(str, Enum):
            VIDEO = "video"

        class TensorlessTorch:
            @staticmethod
            def is_tensor(_value):
                return False

        with self.assertRaisesRegex(RuntimeError, "Unsupported Enum"):
            worker._cpu_tree(OtherDataType.VIDEO, TensorlessTorch)

    def test_cosmos_enum_marker_rejects_unknown_member(self):
        class TensorlessTorch:
            @staticmethod
            def is_tensor(_value):
                return False

        marker = {worker.COSMOS_ENUM_MARKER: {
            "module": "cosmos_policy._src.predict2.conditioner",
            "qualname": "DataType", "member": "UNKNOWN"}}
        with fake_cosmos_data_type():
            with self.assertRaisesRegex(RuntimeError, "Unknown Cosmos Enum member"):
                worker._device_tree(marker, "cuda:0", TensorlessTorch)

    def test_module_import_does_not_import_torch(self):
        self.assertNotIn("torch", sys.modules)

    def test_ours_optimized_dit_scope_temporarily_uses_raw_forward(self):
        def raw_forward(*_args, **_kwargs):
            return "raw"

        def optimized_wrapper(*_args, **_kwargs):
            return "optimized wrapper"

        net = SimpleNamespace(forward=optimized_wrapper)
        graphs = SimpleNamespace(original=raw_forward, records={"stale": object()})
        runtime = SimpleNamespace(model=SimpleNamespace(net=net), optimized_vq_graphs=graphs)
        with worker._dit_replay_dispatch(runtime) as dispatch:
            self.assertIs(net.forward, raw_forward)
            self.assertEqual(graphs.records, {})
            self.assertIn("all-NFE/Cross-KV scheduling bypassed", dispatch)
        self.assertIs(net.forward, optimized_wrapper)
        self.assertEqual(graphs.records, {})

    def test_dit_scope_without_engine_wrapper_uses_generic_dispatch(self):
        runtime = SimpleNamespace(model=SimpleNamespace(net=SimpleNamespace()))
        with worker._dit_replay_dispatch(runtime) as dispatch:
            self.assertEqual(dispatch, "model.net.__call__ generic CUDA graphs")

    def test_shared_pool_graphs_keep_outputs_independent_and_replay_in_nfe_order(self):
        class FakeTensor:
            shape = (1,)
            dtype = "float32"
            device = "cuda:0"

            def __init__(self, value, cuda):
                self.value, self.cuda = value, cuda

            def stride(self):
                return (1,)

            def to(self, **_kwargs):
                return FakeTensor(self.value, self.cuda)

            def copy_(self, source, **_kwargs):
                self.value = source.value
                graph = self.cuda.active_graph
                if graph is not None:
                    destination = self
                    graph.operations.append(
                        lambda dst=destination, src=source: setattr(dst, "value", src.value))
                return self

        class FakeGraph:
            def __init__(self, ordinal):
                self.ordinal, self.pool, self.operations = ordinal, None, []

            def replay(self):
                self.cuda.replay_order.append(self.ordinal)
                for operation in self.operations:
                    operation()

        class FakeStream:
            def wait_stream(self, _stream):
                pass

        class FakeEvent:
            def record(self):
                pass

            def elapsed_time(self, _other):
                return 1.0

        class FakeCuda:
            def __init__(self):
                self.active_graph = None
                self.replay_order = []
                self.graphs = []
                self.shared_pool = object()

            def graph_pool_handle(self):
                return self.shared_pool

            def Stream(self, **_kwargs):
                return FakeStream()

            def current_stream(self, _device):
                return FakeStream()

            @contextmanager
            def stream(self, _stream):
                yield

            def CUDAGraph(self):
                graph = FakeGraph(len(self.graphs))
                graph.cuda = self
                self.graphs.append(graph)
                return graph

            @contextmanager
            def graph(self, graph, *, pool):
                graph.pool = pool
                self.active_graph = graph
                try:
                    yield
                finally:
                    self.active_graph = None

            def synchronize(self, _device=None):
                pass

            def reset_peak_memory_stats(self, _device):
                pass

            def max_memory_allocated(self, _device):
                return 1024

            def Event(self, **_kwargs):
                return FakeEvent()

        class FakeTorch:
            preserve_format = object()

            def __init__(self):
                self.cuda = FakeCuda()

            @staticmethod
            def is_tensor(value):
                return isinstance(value, FakeTensor)

            @staticmethod
            def inference_mode():
                return nullcontext()

            @staticmethod
            def empty_like(value, **_kwargs):
                return FakeTensor(value.value, value.cuda)

        torch = FakeTorch()

        class FakeNet:
            def __call__(self, x):
                return x

        captures = [{"args": (), "kwargs": {"x": FakeTensor(nfe, torch.cuda)}}
                    for nfe in range(5)]
        bundles = worker._make_graph_bundle(FakeNet(), captures, "cuda:0", torch)
        self.assertEqual(len({id(bundle["output"]) for bundle in bundles}), 5)
        self.assertTrue(all(graph.pool is torch.cuda.shared_pool for graph in torch.cuda.graphs))

        replay_calls = [{"args": (), "kwargs": {"x": FakeTensor(10 + nfe, torch.cuda)}}
                        for nfe in range(5)]
        worker._replay_five(bundles, replay_calls, "cuda:0", torch)
        self.assertEqual(torch.cuda.replay_order, list(range(5)))
        self.assertEqual([bundle["output"].value for bundle in bundles], list(range(10, 15)))

    def test_native_extension_build_producer_is_part_of_gate_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "results" / "preparation" / "quarot_fused_build.json"
            path.parent.mkdir(parents=True)
            receipt = {"status": "complete", "pbs_job_id": "25975310.pbs101",
                       "module": "/scratch/native_extensions/quarot_official_gemm_fused.so",
                       "source_commit": "quarot-source",
                       "submodules": {"cutlass": {"commit": "cutlass-commit"}}}
            path.write_text(json.dumps(receipt), encoding="utf-8")
            metadata = {"native_int4_kernel_used": True,
                        "native_backend_module": "quarot_official_gemm_fused",
                        "adapter_version": "cosmos_rotation_int4_g128_v2"}
            binding = worker._binding(metadata, "quarot_w4a4", "current", root)
            producer = binding["extension_build"]
            self.assertEqual(producer["pbs_job_id"], "25975310.pbs101")
            self.assertEqual(producer["source_commit"], "quarot-source")
            self.assertEqual(producer["submodule_commits"], {"cutlass": "cutlass-commit"})

    def test_diffusion_binding_selects_the_requested_arm_build(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "results" / "preparation" / "native_diffusion_build.json"
            path.parent.mkdir(parents=True)
            receipt = {"status": "complete", "pbs_job_id": "25975400.pbs101",
                       "arms": {
                           "svdquant_w4a4": {"module": "/build/svd.so", "source_commit": "svd-commit"},
                           "viditq_w4a4": {"module": "/build/viditq.so", "source_commit": "viditq-commit"}}}
            path.write_text(json.dumps(receipt), encoding="utf-8")
            metadata = {"native_int4_kernel_used": True,
                        "native_extension": "/runtime/viditq.so",
                        "adapter_version": "cosmos_viditq_atom_w4a4_g128_v1"}
            binding = worker._binding(metadata, "viditq_w4a4", "current", root)
            self.assertEqual(binding["extension_build"]["module"], "/build/viditq.so")
            self.assertEqual(binding["extension_build"]["source_commit"], "viditq-commit")
            self.assertEqual(binding["extension_build"]["pbs_job_id"], "25975400.pbs101")

    def test_fused_svd_binding_uses_its_separate_build_producer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'results/preparation/native_svd_fused_build.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'status': 'complete', 'pbs_job_id': 'fused-build',
                'arms': {'svdquant_w4a4_fused_candidate': {
                    'module': '/build/nunchaku_svdq_fused_native.so', 'source_commit': 'fused-source'}}}))
            binding = worker._binding({'native_int4_kernel_used': True,
                'native_extension': '/build/nunchaku_svdq_fused_native.so'},
                'svdquant_w4a4', 'native_fused', root)
            self.assertEqual(binding['extension_build']['pbs_job_id'], 'fused-build')

    def test_offload_finds_plain_tokenizer_model_and_registered_vae(self):
        class FakeDevice:
            def __init__(self, kind):
                self.type = kind

            def __eq__(self, other):
                return isinstance(other, FakeDevice) and self.type == other.type

            def __str__(self):
                return self.type

        cuda, cpu = FakeDevice("cuda"), FakeDevice("cpu")

        class FakeStorage:
            def __init__(self, tensor):
                self.tensor = tensor

            def data_ptr(self):
                return id(self.tensor)

            def nbytes(self):
                return self.tensor.nbytes

        class FakeTensor:
            def __init__(self, nbytes, device):
                self.nbytes, self.device = nbytes, device

            def untyped_storage(self):
                return FakeStorage(self)

            def numel(self):
                return self.nbytes

            def element_size(self):
                return 1

        class FakeModule:
            def __init__(self, parameters=(), children=None):
                self._parameters = list(parameters)
                self._buffers = []
                self._children = dict(children or {})
                self.to_calls = []

            def named_children(self):
                return iter(self._children.items())

            def modules(self):
                yield self
                for child in self._children.values():
                    yield from child.modules()

            def parameters(self, recurse=True):
                yield from self._parameters
                if recurse:
                    for child in self._children.values():
                        yield from child.parameters(recurse=True)

            def buffers(self, recurse=True):
                yield from self._buffers
                if recurse:
                    for child in self._children.values():
                        yield from child.buffers(recurse=True)

            def to(self, device):
                target = cpu if device == "cpu" else device
                self.to_calls.append(target)
                for tensor in list(self.parameters()) + list(self.buffers()):
                    tensor.device = target
                return self

        class FakeTorch:
            nn = SimpleNamespace(Module=FakeModule)

            @staticmethod
            def is_tensor(value):
                return isinstance(value, FakeTensor)

        net_param = FakeTensor(16, cuda)
        net = FakeModule(parameters=[net_param])
        tokenizer_param = FakeTensor(32, cuda)
        tokenizer_model = FakeModule(parameters=[tokenizer_param])
        vae_param = FakeTensor(64, cuda)
        vae = FakeModule(parameters=[vae_param])
        model = FakeModule(children={"net": net, "vae": vae})
        model.net = net
        model.tokenizer = SimpleNamespace(model=tokenizer_model)
        report = worker._offload_non_dit(SimpleNamespace(model=model), FakeTorch, cuda)

        self.assertEqual(tokenizer_param.device.type, "cpu")
        self.assertEqual(vae_param.device.type, "cpu")
        self.assertEqual(net_param.device.type, "cuda")
        modules = {row["module"]: row for row in report["modules"]}
        self.assertEqual(modules["tokenizer.model"]["cuda_tensor_bytes"], 32)
        self.assertEqual(modules["vae"]["cuda_tensor_bytes"], 64)
        self.assertEqual(modules["tokenizer.model"]["remaining_cuda_tensor_bytes"], 0)
        self.assertEqual(report["offloaded_cuda_tensor_bytes"], 96)

    def test_offload_refuses_module_sharing_a_net_parameter(self):
        class FakeDevice:
            def __init__(self, kind):
                self.type = kind

            def __eq__(self, other):
                return isinstance(other, FakeDevice) and self.type == other.type

            def __str__(self):
                return self.type

        cuda = FakeDevice("cuda")

        class FakeStorage:
            def __init__(self, tensor):
                self.tensor = tensor

            def data_ptr(self):
                return id(self.tensor)

            def nbytes(self):
                return 8

        class FakeTensor:
            device = cuda

            def untyped_storage(self):
                return FakeStorage(self)

            def numel(self):
                return 8

            def element_size(self):
                return 1

        class FakeModule:
            def __init__(self, parameters=(), children=None):
                self._parameters = list(parameters)
                self._buffers = []
                self._children = dict(children or {})
                self.to_calls = []

            def named_children(self):
                return iter(self._children.items())

            def modules(self):
                yield self
                for child in self._children.values():
                    yield from child.modules()

            def parameters(self, recurse=True):
                yield from self._parameters
                if recurse:
                    for child in self._children.values():
                        yield from child.parameters(recurse=True)

            def buffers(self, recurse=True):
                yield from self._buffers
                if recurse:
                    for child in self._children.values():
                        yield from child.buffers(recurse=True)

            def to(self, device):
                self.to_calls.append(device)
                return self

        class FakeTorch:
            nn = SimpleNamespace(Module=FakeModule)

            @staticmethod
            def is_tensor(value):
                return isinstance(value, FakeTensor)

        shared = FakeTensor()
        net = FakeModule(parameters=[shared])
        alias = FakeModule(parameters=[shared])
        model = FakeModule(children={"net": net})
        model.net = net
        model.tokenizer = SimpleNamespace(model=alias)
        report = worker._offload_non_dit(SimpleNamespace(model=model), FakeTorch, cuda)

        self.assertEqual(alias.to_calls, [])
        row = next(row for row in report["modules"] if row["module"] == "tokenizer.model")
        self.assertEqual(row["status"], "kept_shared_with_model_net")
        self.assertEqual(row["remaining_cuda_tensor_bytes"], 8)


if __name__ == "__main__":
    unittest.main()
