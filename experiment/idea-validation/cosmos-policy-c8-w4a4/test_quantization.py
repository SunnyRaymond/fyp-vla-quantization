"""Small CPU checks for Cosmos quantization scope, storage, and VQ STE gradients."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch

import quantization
import endpoint


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(129, 3, bias=True, dtype=torch.bfloat16)
        self.k_proj = torch.nn.Linear(129, 3, bias=True, dtype=torch.bfloat16)
        self.v_proj = torch.nn.Linear(129, 3, bias=True, dtype=torch.bfloat16)
        self.output_proj = torch.nn.Linear(129, 3, bias=True, dtype=torch.bfloat16)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.cross_attn = _Attention()
        self.mlp = torch.nn.Module()
        self.mlp.fc1 = torch.nn.Linear(37, 2, bias=True, dtype=torch.bfloat16)
        self.mlp.fc2 = torch.nn.Linear(37, 2, bias=True, dtype=torch.bfloat16)
        self.adaLN_modulation = torch.nn.Linear(37, 2, bias=True, dtype=torch.bfloat16)


class _DiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block()])
        self.patch_embed = torch.nn.Linear(3, 4, dtype=torch.bfloat16)
        self.final_layer = torch.nn.Linear(4, 3, dtype=torch.bfloat16)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _DiT()
        # A tokenizer/VAE-like parameter must stay outside whole-DiT accounting.
        self.tokenizer = torch.nn.Linear(5, 7, dtype=torch.float32)


def _encoding(micro, shape):
    rows, width = shape
    return {
        "format": micro.FORMAT,
        "shape": shape,
        "block_size": 4,
        "scale_block": 32,
        "books": torch.randn(2, 256, 4).to(torch.bfloat16),
        "row_scales": torch.full((rows,), 0.5, dtype=torch.bfloat16),
        "indices": torch.randint(0, 256, (rows, (width + 3) // 4, 2), dtype=torch.uint8),
        "scale_exponents": torch.full((rows, (width + 31) // 32), 127, dtype=torch.uint8),
    }


class QuantizationTests(unittest.TestCase):
    def test_scope_keeps_attention_output_proj_and_skips_final_head(self):
        model = _Model()
        modules = quantization.select_target_linears(model)
        expected = {
            f"net.blocks.0.{attention_name}.{projection}"
            for attention_name in ("self_attn", "cross_attn")
            for projection in ("q_proj", "k_proj", "v_proj", "output_proj")
        } | {"net.blocks.0.mlp.fc1", "net.blocks.0.mlp.fc2"}
        self.assertEqual(set(modules), expected)
        for attention_name in ("self_attn", "cross_attn"):
            for projection in ("q_proj", "k_proj", "v_proj", "output_proj"):
                with self.subTest(attention=attention_name, projection=projection):
                    self.assertIn(f"net.blocks.0.{attention_name}.{projection}", modules)
        self.assertIn("net.blocks.0.mlp.fc1", modules)
        self.assertIn("net.blocks.0.mlp.fc2", modules)
        self.assertNotIn("net.blocks.0.adaLN_modulation", modules)
        self.assertNotIn("net.patch_embed", modules)
        self.assertNotIn("net.final_layer", modules)

    def test_effective_bpw_matches_tensor_payloads_and_model_net_scope(self):
        rotation, micro, _, _, _ = quantization._helpers()
        model = _Model()
        modules = quantization.select_target_linears(model)
        report = quantization.effective_bpw(modules, model)

        packed_bytes = scale_bytes = 0
        aliases = {"video_expert." + name: module for name, module in modules.items()}
        state = rotation.build_rotation_state(aliases, mode="learned", seed=23)
        for module in modules.values():
            codes, scales = rotation._weight_codes_and_scales(module.weight)
            packed = rotation._pack_nibbles(codes)
            packed_bytes += packed.numel() * packed.element_size()
            scale_bytes += scales.numel() * scales.element_size()
        stream = state.streams["video"]
        rotation_bytes = (state.matrix("video").numel() * state.matrix("video").element_size()
                          + stream.signs.numel() * stream.signs.element_size())
        self.assertEqual(report["spinquant"]["bytes"], packed_bytes + scale_bytes + rotation_bytes)

        vq_payload_bytes = 0
        for module in modules.values():
            encoding = _encoding(micro, (module.out_features, module.in_features))
            micro.check_encoding(encoding)
            transform_payload = {
                "scale": torch.ones(module.in_features, dtype=torch.float32),
                "signs": torch.ones(module.in_features, dtype=torch.float32),
            }
            vq_payload_bytes += sum(value.numel() * value.element_size()
                                    for value in encoding.values() if torch.is_tensor(value))
            vq_payload_bytes += sum(value.numel() * value.element_size()
                                    for value in transform_payload.values())
        self.assertEqual(report["micro_vq"]["bytes"], vq_payload_bytes)

        dit_parameters = list(model.net.parameters())
        self.assertEqual(report["whole_dit"]["parameter_numel"],
                         sum(parameter.numel() for parameter in dit_parameters))
        self.assertEqual(report["whole_dit"]["parameter_bytes"],
                         sum(parameter.numel() * parameter.element_size() for parameter in dit_parameters))
        self.assertNotEqual(report["whole_dit"]["parameter_numel"],
                            sum(parameter.numel() for parameter in model.parameters()))

    def test_vq_ste_gradients_are_finite_and_indices_exponents_stay_fixed(self):
        _, micro, _, _, _ = quantization._helpers()
        name = "net.blocks.0.self_attn.output_proj"
        module = torch.nn.Linear(37, 3, bias=True, dtype=torch.bfloat16)
        encoding = _encoding(micro, (3, 37))
        transform = {
            "scale": torch.ones(37, dtype=torch.float32),
            "signs": torch.where(torch.arange(37) % 2 == 0, 1.0, -1.0),
            "hadamard": True,
            "block": 128,
            "input_dim": 37,
        }
        indices_before = encoding["indices"].clone()
        exponents_before = encoding["scale_exponents"].clone()
        state = quantization.vq_train_state({name: module}, {name: encoding}, {name: transform})
        optimizer = torch.optim.SGD(state.parameters(), lr=1e-3)
        values = torch.randn(2, 37, dtype=torch.bfloat16, requires_grad=True)
        loss = module(values).float().square().mean()
        loss.backward()

        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertGreater(float(values.grad.abs().sum()), 0.0)
        for parameter in state.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
        optimizer.step()
        state.refresh()
        self.assertTrue(torch.equal(state.encodings[name]["indices"], indices_before))
        self.assertTrue(torch.equal(state.encodings[name]["scale_exponents"], exponents_before))
        state.restore_forwards()

    def test_spin_wrappers_backprop_under_native_selective_checkpoint(self):
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            checkpoint,
            create_selective_checkpoint_contexts,
        )

        rotation, *_ = quantization._helpers()

        def run_policy(policy_name):
            name = "net.blocks.0.self_attn.q_proj"
            model = _Model()
            module = dict(model.named_modules())[name]
            state, handle, _aliases = quantization.build_spin_training({name: module}, seed=43)
            self.assertFalse(module._rotation_training_checkpoint)
            self.assertTrue(state._matrix_cache, "R128 must be warmed outside the block checkpoint")
            counts = {"forward": 0, "recompute": 0}

            def selective_mm_policy(context, function, *_args, **_kwargs):
                if function == torch.ops.aten.mm.default:
                    if policy_name == "save_all_mm":
                        return CheckpointPolicy.MUST_SAVE
                    mode = "recompute" if context.is_recompute else "forward"
                    counts[mode] = (counts[mode] + 1) % 16
                    if counts[mode] > 8:
                        return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE

            def context_fn():
                nonlocal counts
                counts = {"forward": 0, "recompute": 0}
                return create_selective_checkpoint_contexts(selective_mm_policy)

            identity = torch.eye(module.in_features, dtype=torch.bfloat16)
            optimizer = torch.optim.SGD(
                [parameter for parameter in state.parameters() if parameter.requires_grad], lr=1e-3
            )

            def native_block(input_value):
                projected = module(input_value)
                replayed = input_value.reshape(-1, module.in_features)
                for _ in range(16):
                    replayed = torch.mm(replayed, identity)
                return projected + replayed[:, :module.out_features].reshape_as(projected)

            try:
                for _update in range(2):
                    self.assertTrue(state._matrix_cache)
                    values = torch.randn(1, 5, module.in_features, dtype=torch.bfloat16,
                                         requires_grad=True)
                    output = checkpoint(native_block, values, use_reentrant=False,
                                        context_fn=context_fn)
                    output.float().square().mean().backward()
                    self.assertIsNotNone(values.grad)
                    self.assertTrue(torch.isfinite(values.grad).all())
                    rotation_grads = [parameter.grad for parameter in state.parameters()]
                    self.assertTrue(rotation_grads)
                    for gradient in rotation_grads:
                        self.assertIsNotNone(gradient)
                        self.assertTrue(torch.isfinite(gradient).all())
                        self.assertGreater(float(gradient.abs().sum()), 0.0)
                    old_matrix = state.matrix("video")
                    handle.optimizer_step(optimizer)
                    self.assertFalse(state._matrix_cache, "optimizer refresh must drop the old Cayley graph")
                    quantization.prewarm_spin_rotations(state)
                    self.assertTrue(state._matrix_cache, "next block replay must have a prebuilt R128")
                    self.assertIsNot(state.matrix("video"), old_matrix)
            finally:
                rotation.restore_original_linears(handle)

        for policy_name in ("save_all_mm", "predict2_2b_720_counted_mm"):
            with self.subTest(policy=policy_name):
                run_policy(policy_name)

    def test_endpoint_motor_loss_excludes_gripper_and_responds_to_motor_error(self):
        def result(action):
            zero = torch.zeros(1, 2, 2, 2)
            return endpoint.EndpointResult(
                generated_latent=None, orig_clean_latent_frames=None, action=action,
                future_image_slots={"future_wrist": zero, "future_image": zero.clone()},
                sampler_evaluations=5, denoiser_calls=5, packets=[],
            )

        teacher_action = torch.zeros(1, 16, 7)
        teacher = result(teacher_action)
        same_motors_changed_gripper = teacher_action.clone()
        same_motors_changed_gripper[..., 6] = 3.0
        gripper_changed = result(same_motors_changed_gripper)
        base_motor_loss, _ = endpoint.endpoint_losses(
            teacher, teacher, action_motor_dimensions=6)
        gripper_motor_loss, _ = endpoint.endpoint_losses(
            gripper_changed, teacher, action_motor_dimensions=6)
        self.assertEqual(float(base_motor_loss), 0.0)
        self.assertEqual(float(gripper_motor_loss), 0.0)
        self.assertGreater(float(endpoint.endpoint_gripper_mse(gripper_changed, teacher)), 0.0)

        changed_motor = teacher_action.clone()
        changed_motor[..., 2] = 1.0
        motor_loss, _ = endpoint.endpoint_losses(
            result(changed_motor), teacher, action_motor_dimensions=6)
        self.assertGreater(float(motor_loss), 0.0)

    def _make_spin_artifact(self, root, model, modules, *, seed=31):
        rotation, *_ = quantization._helpers()
        state, handle, aliases = quantization.build_spin_training(modules, seed=seed)
        try:
            for name, module in modules.items():
                self.assertTrue(torch.isfinite(module.weight).all().item(),
                                msg=f"seed={seed} nonfinite base weight at {name}")
                if module.bias is not None:
                    self.assertTrue(torch.isfinite(module.bias).all().item(),
                                    msg=f"seed={seed} nonfinite base bias at {name}")
            report = quantization.effective_bpw(modules, model)
            metadata = quantization.save_spin_deployment(
                root, modules, state, aliases, report, seed=seed,
            )
            self.assertEqual(metadata["training_vs_deployed_probe"]["deployed_weight_operand_dtype"],
                             "torch.bfloat16", msg=f"seed={seed}")
            self.assertEqual(metadata["training_vs_deployed_probe"]["deployed_activation_operand_dtype"],
                             "torch.bfloat16", msg=f"seed={seed}")
        finally:
            rotation.restore_original_linears(handle)

    def _make_vq_artifact(self, root, model, modules):
        _, micro, _, _, _ = quantization._helpers()
        encodings, transforms = {}, {}
        for name, module in modules.items():
            rows, width = module.weight.shape
            generator = torch.Generator(device="cpu").manual_seed(23)
            encodings[name] = {
                "format": micro.FORMAT, "shape": (rows, width), "block_size": 4,
                "scale_block": 32,
                "books": torch.randn((2, 256, 4), generator=generator).to(torch.bfloat16),
                "row_scales": torch.full((rows,), 0.5, dtype=torch.bfloat16),
                "indices": torch.randint(0, 256, (rows, (width + 3) // 4, 2),
                                          generator=generator, dtype=torch.uint8),
                "scale_exponents": torch.full((rows, (width + 31) // 32), 127,
                                              dtype=torch.uint8),
            }
            transforms[name] = {
                "scale": torch.ones(width, dtype=torch.float32),
                "signs": torch.where(torch.arange(width) % 2 == 0, 1.0, -1.0),
                "hadamard": True, "block": 128, "input_dim": width,
            }

        class _BankPayload:
            def deployment(self):
                return {name: {"encoding": encodings[name], "transform": transforms[name]}
                        for name in modules}

        report = quantization.effective_bpw(modules, model)
        quantization.save_vq_deployment(root, modules, _BankPayload(), report)

    def _assert_cached_bf16_inference(self, arm, artifact_root, model, *, seed=31):
        name = "net.blocks.0.self_attn.q_proj"
        module = dict(model.named_modules())[name]
        controller = quantization.QuantizationController(model, {name: module}, root=artifact_root)
        torch_load = torch.load
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 100003)
        x = torch.randn(2, module.in_features, generator=generator, dtype=torch.bfloat16)
        try:
            self.assertTrue(torch.isfinite(x).all().item(), msg=f"seed={seed} nonfinite input")
            with mock.patch.object(torch, "load", wraps=torch_load) as load_spy:
                controller.set_arm(arm, artifact_dir=artifact_root)
                loads_after_first_install = load_spy.call_count
                self.assertGreater(loads_after_first_install, 0, msg=f"seed={seed}")
                decoded_weight = module.weight.detach().clone()
                self.assertTrue(torch.isfinite(decoded_weight).all().item(),
                                msg=f"seed={seed} nonfinite decoded weight")
                if arm == quantization.SPIN_ARM:
                    state = module.__dict__["_cosmos_spin_state"]
                    rotated = state["rotate"](x, state["matrix"], state["signs"])
                    qx = state["a4_dequant"](rotated).to(torch.bfloat16)
                else:
                    transform = controller._vq_state.transforms[name]
                    bank = controller._vq_state.train_banks
                    qx = bank.A4Input.apply(x, transform)
                reference = torch.nn.functional.linear(qx, decoded_weight, module.bias)
                observed = []
                original_linear = torch.nn.functional.linear

                def record_operands(input_value, weight, bias=None):
                    observed.append((input_value.dtype, weight.dtype,
                                     None if bias is None else bias.dtype))
                    return original_linear(input_value, weight, bias)

                with mock.patch.object(torch.nn.functional, "linear", side_effect=record_operands):
                    output = module(x)
                self.assertEqual(output.dtype, torch.bfloat16, msg=f"seed={seed}")
                self.assertEqual(observed, [(torch.bfloat16, torch.bfloat16, torch.bfloat16)],
                                 msg=f"seed={seed}")
                self.assertTrue(torch.equal(output, reference), msg=f"seed={seed}")

                controller.set_arm(arm, artifact_dir=artifact_root)
                self.assertEqual(load_spy.call_count, loads_after_first_install, msg=f"seed={seed}")
                self.assertTrue(torch.equal(module(x), reference), msg=f"seed={seed}")

                controller.set_arm(quantization.BF16_ARM)
                controller.set_arm(arm, artifact_dir=artifact_root)
                self.assertEqual(load_spy.call_count, loads_after_first_install, msg=f"seed={seed}")
                self.assertTrue(torch.equal(module(x), reference), msg=f"seed={seed}")
        finally:
            controller.close()

    def test_spin_inference_uses_exported_bank_bf16_and_host_cache(self):
        name = "net.blocks.0.self_attn.q_proj"
        for seed in (1, 7, 31, 127, 20261015):
            with self.subTest(seed=seed), TemporaryDirectory() as temporary:
                torch.manual_seed(seed)
                model = _Model()
                module = dict(model.named_modules())[name]
                root = Path(temporary)
                self._make_spin_artifact(root, model, {name: module}, seed=seed)
                self._assert_cached_bf16_inference(
                    quantization.SPIN_ARM, root, model, seed=seed,
                )

    def test_vq_inference_uses_fixed_decoded_bf16_and_host_cache(self):
        with TemporaryDirectory() as temporary:
            model = _Model()
            name = "net.blocks.0.self_attn.q_proj"
            module = dict(model.named_modules())[name]
            root = Path(temporary)
            self._make_vq_artifact(root, model, {name: module})
            self._assert_cached_bf16_inference(quantization.VQ_ARM, root, model)


if __name__ == "__main__":
    unittest.main()
