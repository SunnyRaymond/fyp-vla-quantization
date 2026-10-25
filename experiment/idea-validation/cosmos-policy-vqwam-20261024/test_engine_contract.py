"""CPU-only metadata checks; does not import torch or validate GPU execution."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import unittest
from enum import Enum
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import engine


class ManifestContractTests(unittest.TestCase):
    @staticmethod
    def payload(count=engine.EXPECTED_TARGETS):
        modules = {
            f"net.blocks.{index:03d}.attn.to_q": SimpleNamespace(out_features=2, in_features=3)
            for index in range(count)
        }
        rows = engine._target_rows(modules)
        manifest = {
            "format": engine.FORMAT,
            "protocol": "cosmos_vqwam_20261024",
            "source_job": "26000000.pbs101",
            "completed_updates": 8,
            "target_modules": rows,
            "modules": [{"name": row["name"], "shape": row["shape"],
                         "file": f"bank/{index:04d}.pt"}
                        for index, row in enumerate(rows)],
            "rotation_file": "rotation.pt",
        }
        return modules, manifest

    def test_sorted_280_bank_contract_is_accepted(self):
        modules, manifest = self.payload()
        self.assertEqual(len(engine.validate_manifest(
            manifest, modules, expected_protocol="cosmos_vqwam_20261024")), 280)

    def test_target_order_mismatch_is_rejected(self):
        modules, manifest = self.payload()
        manifest["target_modules"].reverse()
        with self.assertRaisesRegex(ValueError, "target_modules"):
            engine.validate_manifest(manifest, modules, expected_protocol="cosmos_vqwam_20261024")

    def test_bank_path_traversal_is_rejected(self):
        modules, manifest = self.payload()
        manifest["modules"][0]["file"] = "bank/../0000.pt"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            engine.validate_manifest(manifest, modules, expected_protocol="cosmos_vqwam_20261024")

    def test_protocol_mismatch_is_rejected(self):
        modules, manifest = self.payload()
        with self.assertRaisesRegex(ValueError, "protocol"):
            engine.validate_manifest(manifest, modules, expected_protocol="another_protocol")


class DecoderReceiptTests(unittest.TestCase):
    @staticmethod
    def payload():
        modules = {
            f"net.blocks.{index:03d}.attn.to_q": SimpleNamespace(out_features=2, in_features=3)
            for index in range(engine.EXPECTED_TARGETS)
        }
        manifest = {"protocol": "cosmos_vqwam_20261024", "source_job": "26000000.pbs101"}
        rows = engine._target_rows(modules)
        receipt = {
            "format": engine.DECODER_RECEIPT_FORMAT,
            "status": "passed",
            "arm": engine.ARM,
            "protocol": manifest["protocol"],
            "source_job": manifest["source_job"],
            "device": {"name": "GPU-A", "capability": [8, 0]},
            "shapes": [{"shape": [2, 3], "module_count": engine.EXPECTED_TARGETS,
                        "selected": "reference"}],
            "modules": [{"name": row["name"], "shape": row["shape"],
                         "reference_bitwise": True, "selected": "reference",
                         "selected_bitwise": True} for row in rows],
            "parity": {"reference_checked_modules": engine.EXPECTED_TARGETS,
                       "reference_bitwise_passed": engine.EXPECTED_TARGETS,
                       "selected_checked_modules": engine.EXPECTED_TARGETS,
                       "selected_bitwise_passed": engine.EXPECTED_TARGETS,
                       "failed_modules": []},
        }
        return modules, manifest, receipt

    def test_complete_receipt_resolves_per_shape_choice(self):
        modules, manifest, receipt = self.payload()
        self.assertEqual(engine.validate_decoder_receipt(receipt, manifest, modules),
                         {(2, 3): "reference"})

    def test_incomplete_module_parity_is_rejected(self):
        modules, manifest, receipt = self.payload()
        receipt["modules"].pop()
        with self.assertRaisesRegex(ValueError, "every target"):
            engine.validate_decoder_receipt(receipt, manifest, modules)

    def test_receipt_for_another_arm_is_rejected(self):
        modules, manifest, receipt = self.payload()
        receipt["arm"] = "bf16"
        with self.assertRaisesRegex(ValueError, "arm/protocol"):
            engine.validate_decoder_receipt(receipt, manifest, modules)

    def test_receipt_tuned_for_another_gpu_is_rejected(self):
        modules, manifest, receipt = self.payload()
        with self.assertRaisesRegex(ValueError, "different GPU"):
            engine.validate_decoder_receipt(receipt, manifest, modules,
                                            expected_device={"name": "GPU-B", "capability": [8, 0]})


class Full80ParityTests(unittest.TestCase):
    @staticmethod
    def payload(checked=80, expected=80, failed=None, status="passed"):
        manifest = {"protocol": "cosmos_vqwam_20261024", "source_job": "26000000.pbs101",
                    "completed_updates": 8}
        receipt = {"status": status, "protocol": manifest["protocol"],
                   "source_job": manifest["source_job"], "checked_queries": checked,
                   "expected_queries": expected, "failed_queries": [] if failed is None else failed}
        return manifest, receipt

    @staticmethod
    def fake_engine(graph_replays=0, capture_errors=None):
        class Graphs:
            graph_status = "pending_graph_validation"

            def __init__(self):
                self.graph_replays = graph_replays
                self.capture_errors = [] if capture_errors is None else list(capture_errors)

            def metadata(self):
                return {"status": self.graph_status, "graph_replays": self.graph_replays,
                        "capture_errors": list(self.capture_errors)}

        item = engine.Engine.__new__(engine.Engine)
        item.manifest, item.parity_receipt = Full80ParityTests.payload()[0], None
        item.graphs = Graphs()
        item.runtime = SimpleNamespace()
        item.banks = SimpleNamespace(decoder_receipt=None, modules={
            f"net.blocks.{index:03d}.attn.to_q": SimpleNamespace(out_features=2, in_features=3)
            for index in range(engine.EXPECTED_TARGETS)})
        item.artifact_dir = "artifact"
        item.deployment_tensor_bytes = item.unquantized_dit_tensor_bytes = 1
        item.cache = SimpleNamespace(metadata=lambda: {})
        item._workspace_metadata = lambda: {}
        return item

    def test_incomplete_receipt_stays_pending(self):
        manifest, receipt = self.payload(checked=79)
        item = self.fake_engine(graph_replays=1)
        item.manifest = manifest
        self.assertEqual(engine.Engine.metadata(item)["parity_status"],
                         "pending_real_GPU_reference_comparison")
        with self.assertRaisesRegex(ValueError, "exactly 80"):
            item.mark_validated(receipt)
        self.assertIsNone(item.parity_receipt)
        self.assertEqual(item.graphs.graph_status, "pending_graph_validation")

    def test_receipt_requires_current_process_graph_replay(self):
        manifest, receipt = self.payload()
        item = self.fake_engine(graph_replays=0)
        item.manifest = manifest
        with self.assertRaisesRegex(ValueError, "clean CUDA graph replay"):
            item.mark_validated(receipt)
        self.assertEqual(engine.Engine.metadata(item)["parity_status"],
                         "pending_real_GPU_reference_comparison")

    def test_failed_query_receipt_is_rejected(self):
        manifest, receipt = self.payload(failed=[17])
        item = self.fake_engine(graph_replays=1)
        item.manifest = manifest
        with self.assertRaisesRegex(ValueError, "failed queries"):
            item.mark_validated(receipt)
        self.assertIsNone(item.parity_receipt)

    def test_nonpassed_or_wrong_artifact_receipt_is_rejected(self):
        manifest, receipt = self.payload(status="running")
        with self.assertRaisesRegex(ValueError, "did not pass"):
            engine.validate_full80_parity_receipt(receipt, manifest)
        _, receipt = self.payload()
        receipt["source_job"] = "different.pbs"
        with self.assertRaisesRegex(ValueError, "different artifact"):
            engine.validate_full80_parity_receipt(receipt, manifest)

    def test_capture_errors_block_validation(self):
        manifest, receipt = self.payload()
        item = self.fake_engine(graph_replays=1, capture_errors=["graph capture failed"])
        item.manifest = manifest
        with self.assertRaisesRegex(ValueError, "clean CUDA graph replay"):
            item.mark_validated(receipt)
        self.assertEqual(item.graphs.graph_status, "pending_graph_validation")

    def test_full80_receipt_marks_metadata_only_after_clean_graph_replay(self):
        manifest, receipt = self.payload()
        item = self.fake_engine(graph_replays=1)
        item.manifest = manifest
        metadata = item.mark_validated(receipt)
        self.assertEqual(metadata["parity_status"], "validated_full80")
        self.assertEqual(metadata["cuda_graph"]["status"], "validated_full80")
        self.assertIn("full 80-query", metadata["execution_scope"])
        self.assertIs(item.parity_receipt, receipt)
        item.graphs.capture_errors.append("capture failed")
        item.graphs.graph_status = "pending_graph_validation"
        self.assertEqual(engine.Engine.metadata(item)["parity_status"],
                         "pending_real_GPU_reference_comparison")


class GraphCaptureCacheTests(unittest.TestCase):
    @staticmethod
    def cache():
        cache = engine._CrossKVCache.__new__(engine._CrossKVCache)
        cache.active = True
        cache.graph_capture_active = False
        cache.hits = engine.Counter()
        cache._same_input = lambda *_args: (_ for _ in ()).throw(AssertionError("equality check ran"))
        return cache

    @staticmethod
    def slot(*, cacheable=True, valid=True, matches=1, buffer=None, original=None):
        calls = []

        def run(value, *args, **kwargs):
            calls.append(value)
            return "original"

        return {"original": run if original is None else original,
                "cacheable": cacheable, "valid": valid, "matches": matches,
                "calls": 0, "buffer": "cached" if buffer is None else buffer,
                "input": None, "version": None, "snapshot": None, "buffers": {}}, calls

    def test_capture_scope_reuses_only_confirmed_slot_without_tensor_equal(self):
        cache = self.cache()
        slot, calls = self.slot()
        with cache.capture_scope():
            result = engine._CrossKVCache._forward(cache, "cross.attn.k", slot, object())
        self.assertEqual(result, "cached")
        self.assertEqual(calls, [])
        self.assertEqual(cache.hits["cross.attn.k"], 1)
        self.assertFalse(cache.graph_capture_active)

    def test_capture_keeps_dynamic_and_unconfirmed_slots_eager(self):
        cache = self.cache()
        with cache.capture_scope():
            for kwargs in ({"cacheable": False}, {"valid": False, "matches": 0}):
                slot, calls = self.slot(**kwargs)
                self.assertEqual(engine._CrossKVCache._forward(cache, "cross.attn.v", slot, object()),
                                 "original")
                self.assertEqual(len(calls), 1)

    def test_capture_flag_is_cleared_after_exception_and_eager_validation_resumes(self):
        cache = self.cache()
        with self.assertRaisesRegex(RuntimeError, "capture failure"):
            with cache.capture_scope():
                raise RuntimeError("capture failure")
        self.assertFalse(cache.graph_capture_active)
        cache._same_input = lambda *_args: False
        slot, calls = self.slot()
        self.assertEqual(engine._CrossKVCache._forward(cache, "cross.attn.k", slot, object()),
                         "original")
        self.assertEqual(len(calls), 1)


class FakeTensor:
    clone_count = 0

    def __init__(self, value, pointer, version=0, inference=False):
        self.value, self.pointer, self.version, self.inference = value, pointer, version, inference
        self.shape, self.dtype, self.device = (1,), "fake-bf16", "cuda:0"

    @property
    def _version(self):
        if self.inference:
            raise RuntimeError("inference tensor has no version counter")
        return self.version

    def data_ptr(self):
        return self.pointer

    def detach(self):
        return self

    def clone(self, memory_format=None):
        type(self).clone_count += 1
        return FakeTensor(self.value, self.pointer + 10000, self.version)

    def copy_(self, other):
        self.value = other.value
        return self

    def equal(self, other):
        return isinstance(other, FakeTensor) and self.value == other.value


class FakeDType:
    def __init__(self, name):
        self.name = name
        self.self = self

    def __eq__(self, other):
        return isinstance(other, FakeDType) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class DenoiseTreeTraversalTests(unittest.TestCase):
    class Mode(Enum):
        IDLE = 0
        ACTIVE = 1

    @staticmethod
    def fake_torch():
        torch = types.ModuleType("torch")
        torch.Tensor = FakeTensor
        torch.dtype = FakeDType
        torch.preserve_format = object()
        return torch

    def test_net_graph_keeps_native_condition_branch_eager_and_restores_hooks(self):
        torch = self.fake_torch()
        state = {'capturing': False, 'branch_calls': 0, 'net_capture_calls': 0}

        class ConditionFlag:
            value = True

            def __bool__(self):
                if state['capturing']:
                    raise RuntimeError('Tensor truth read inside capture')
                state['branch_calls'] += 1
                return self.value

        @contextmanager
        def capture(_graph):
            state['capturing'] = True
            try:
                yield
            finally:
                state['capturing'] = False

        stream = SimpleNamespace(wait_stream=lambda _other: None)
        torch.cuda = SimpleNamespace(Stream=lambda **_kw: stream,
            current_stream=lambda _device: stream, stream=lambda _stream: nullcontext(),
            synchronize=lambda _device: None, graph=capture,
            CUDAGraph=lambda: SimpleNamespace(replay=lambda: None))

        def net_forward(x_B_C_T_H_W, timesteps_B_T):
            state['net_capture_calls'] += int(state['capturing'])
            return FakeTensor(x_B_C_T_H_W.value + timesteps_B_T.value, 500)

        net = SimpleNamespace(forward=net_forward)

        def denoise(x, sigma, condition):
            use_video = bool(condition.use_video_condition)
            result = net.forward(x_B_C_T_H_W=x, timesteps_B_T=sigma)
            return result.value if use_video else -result.value

        generate = lambda _batch: None
        model = SimpleNamespace(net=net, denoise=denoise,
                                generate_samples_from_batch=generate)
        cache = SimpleNamespace(active=True, modules={}, denoise_calls=0,
                                signature=lambda: (), capture_scope=lambda: nullcontext())
        banks = SimpleNamespace(activate_graph_workspace=lambda _key: 'workspace')
        condition = SimpleNamespace(use_video_condition=ConditionFlag())
        x, sigma = FakeTensor(5, 10), FakeTensor(1, 20)
        with patch.dict(sys.modules, {'torch': torch}):
            graphs = engine._DenoiseGraphs(SimpleNamespace(model=model), banks, cache)
            self.assertEqual([model.denoise(x, sigma, condition) for _ in range(5)], [6] * 5)
            self.assertEqual(state['branch_calls'], 5)
            self.assertEqual(state['net_capture_calls'], 1)
            self.assertEqual(graphs.calls['graph_captures'], 1)
            self.assertEqual(graphs.calls['graph_replays'], 2)
            self.assertFalse(graphs.graph_enabled)
            condition.use_video_condition.value = False
            self.assertEqual(model.denoise(x, sigma, condition), -6)
            self.assertEqual(graphs.calls['condition_changes'], 1)
            self.assertEqual(graphs.calls['graph_replays'], 2)
            graphs.restore()
        self.assertIs(model.denoise, denoise)
        self.assertIs(model.net.forward, net_forward)
        self.assertIs(model.generate_samples_from_batch, generate)

    def test_cycle_shared_alias_enum_and_dtype_are_safe_across_denoise_tree_paths(self):
        torch = self.fake_torch()
        tensor = FakeTensor(5, 101, version=2)
        child = SimpleNamespace(flag=True, mode=self.Mode.IDLE)
        condition = SimpleNamespace(left=child, right=child, tensor=tensor,
                                    mode_type=self.Mode, dtype=FakeDType("bfloat16"))
        condition.self = condition
        child.owner = condition
        with patch.dict(sys.modules, {"torch": torch}):
            versions = engine._DenoiseGraphs._versions(condition)
            signature = engine._signature(torch, condition)
            cloned = engine._clone_tree(torch, condition)
            self.assertIs(cloned.self, cloned)
            self.assertIs(cloned.left, cloned.right)
            self.assertIs(cloned.left.owner, cloned)
            self.assertIsNot(cloned.tensor, tensor)
            self.assertIs(cloned.mode_type, self.Mode)
            self.assertTrue(engine._DenoiseGraphs._same_tree(condition, condition, versions))
            engine._copy_tree(torch, cloned, condition)
            self.assertIs(cloned.left, cloned.right)

            unshared = SimpleNamespace(left=SimpleNamespace(flag=True, mode=self.Mode.IDLE),
                                       right=SimpleNamespace(flag=True, mode=self.Mode.IDLE),
                                       tensor=tensor, mode_type=self.Mode,
                                       dtype=FakeDType("bfloat16"))
            unshared.self = unshared
            unshared.left.owner = unshared
            unshared.right.owner = unshared
            self.assertFalse(engine._DenoiseGraphs._same_tree(unshared, condition, versions))
            self.assertNotEqual(signature, engine._signature(torch, unshared))
            with self.assertRaisesRegex(ValueError, "alias topology changed"):
                engine._copy_tree(torch, cloned, unshared)

            child.flag = False
            self.assertFalse(engine._DenoiseGraphs._same_tree(condition, condition, versions))

    def test_enum_value_changes_are_compared_to_snapshot_not_mutable_reference(self):
        torch = self.fake_torch()
        condition = SimpleNamespace(enabled=True, mode=self.Mode.IDLE)
        with patch.dict(sys.modules, {"torch": torch}):
            versions = engine._DenoiseGraphs._versions(condition)
            self.assertTrue(engine._DenoiseGraphs._same_tree(condition, condition, versions))
            condition.enabled = False
            self.assertFalse(engine._DenoiseGraphs._same_tree(condition, condition, versions))
            condition.enabled = True
            condition.mode = self.Mode.ACTIVE
            self.assertFalse(engine._DenoiseGraphs._same_tree(condition, condition, versions))

    def test_reachable_tensor_pointer_version_and_inference_snapshot_are_preserved(self):
        torch = self.fake_torch()
        tensor = FakeTensor(5, 101, version=2)
        condition = SimpleNamespace(nested={"tensor": tensor})
        with patch.dict(sys.modules, {"torch": torch}):
            original = engine._DenoiseGraphs._versions(condition)
            self.assertTrue(engine._DenoiseGraphs._same_tree(condition, condition, original))
            tensor.version += 1
            self.assertNotEqual(original, engine._DenoiseGraphs._versions(condition))
            self.assertFalse(engine._DenoiseGraphs._same_tree(condition, condition, original))

            tensor.version -= 1
            pointer_signature = engine._DenoiseGraphs._versions(condition)
            tensor.pointer += 1
            self.assertNotEqual(pointer_signature, engine._DenoiseGraphs._versions(condition))

            inference = FakeTensor(9, 301, inference=True)
            inference_condition = SimpleNamespace(nested=[inference])
            inference_versions = engine._DenoiseGraphs._versions(inference_condition)
            snapshot = inference_versions[3][0][1][3][0][2]
            self.assertIsInstance(snapshot, FakeTensor)
            self.assertTrue(engine._DenoiseGraphs._same_tree(
                inference_condition, inference_condition, inference_versions))
            inference.value = 10
            self.assertFalse(engine._DenoiseGraphs._same_tree(
                inference_condition, inference_condition, inference_versions))

    def test_tensor_signature_tracks_alias_topology_but_not_fresh_pointers(self):
        torch = self.fake_torch()
        shared_tensor = FakeTensor(1, 11)
        shared = SimpleNamespace(left=shared_tensor, right=shared_tensor)
        fresh_shared_tensor = FakeTensor(9, 99)
        fresh_shared = SimpleNamespace(left=fresh_shared_tensor, right=fresh_shared_tensor)
        independent = SimpleNamespace(left=FakeTensor(1, 11), right=FakeTensor(1, 22))
        with patch.dict(sys.modules, {"torch": torch}):
            shared_signature = engine._signature(torch, shared)
            self.assertEqual(shared_signature, engine._signature(torch, fresh_shared))
            self.assertNotEqual(shared_signature, engine._signature(torch, independent))

            FakeTensor.clone_count = 0
            cloned = engine._clone_tree(torch, shared)
            self.assertIs(cloned.left, cloned.right)
            self.assertEqual(FakeTensor.clone_count, 1)


if __name__ == "__main__":
    unittest.main()
