"""Bounded teacher-future intervention for Fast-WAM Optional IDM."""
from __future__ import annotations


ACTION_HORIZON = 32
VIDEO_FRAMES = 9
INFERENCE_STEPS = 10
SIGMA_SHIFT = 1.0


def _infer_kwargs(runtime, datum, seed):
    return {
        "prompt": runtime.ev.DEFAULT_PROMPT.format(task=runtime.description),
        "input_image": datum["image"],
        "proprio": datum["proprio"],
        "action_horizon": ACTION_HORIZON,
        "num_video_frames": VIDEO_FRAMES,
        "num_inference_steps": INFERENCE_STEPS,
        "sigma_shift": SIGMA_SHIFT,
        "seed": int(seed),
        "rand_device": "cpu",
        "tiled": False,
        "compile_action_infer": False,
        "action_infer_mode": "idm",
    }


def _expected_latent_shape(model, datum):
    image_shape = tuple(datum["image"].shape)
    if len(image_shape) == 3 and image_shape[0] == 3:
        height, width = image_shape[1:]
    elif len(image_shape) == 4 and image_shape[:2] == (1, 3):
        height, width = image_shape[2:]
    else:
        raise ValueError(f"Expected datum image [3,H,W] or [1,3,H,W], got {image_shape}")
    vae = model.vae
    temporal_factor = int(vae.temporal_downsample_factor)
    if temporal_factor <= 0:
        raise ValueError(f"Invalid VAE temporal_downsample_factor: {temporal_factor}")
    return (
        1,
        int(vae.model.z_dim),
        (VIDEO_FRAMES - 1) // temporal_factor + 1,
        int(height) // int(vae.upsampling_factor),
        int(width) // int(vae.upsampling_factor),
    )


def _patch(obj, name, replacement, patches):
    namespace = getattr(obj, "__dict__", {})
    existed = name in namespace
    previous = namespace.get(name)
    patches.append((obj, name, existed, previous))
    setattr(obj, name, replacement)


def _restore(patches):
    for obj, name, existed, previous in reversed(patches):
        if existed:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


def _run(runtime, datum, seed, teacher_latents=None):
    model = runtime.model
    replacing = teacher_latents is not None
    counts = {
        "video_dit_calls": 0,
        "video_denoise_loop_calls": 0,
        "video_scheduler_steps": 0,
        "action_dit_calls": 0,
        "teacher_future_injection_steps": 0,
    }
    diagnostic = {"action_initial_noise": None, "action_conditioning_latents": None}
    patches = []
    first_frame = {"value": None}

    if replacing:
        expected_shape = _expected_latent_shape(model, datum)
        actual_shape = tuple(int(value) for value in teacher_latents.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"teacher_latents must have shape {expected_shape}, got {actual_shape}"
            )
        teacher = teacher_latents.detach().to(
            device=model.device, dtype=model.torch_dtype
        )

        def skip_video_dit(*, latents_video, **_kwargs):
            counts["video_denoise_loop_calls"] += 1
            return None

        def install_teacher_future(_prediction, _delta, sample):
            counts["video_scheduler_steps"] += 1
            if first_frame["value"] is None:
                first_frame["value"] = sample[:, :, 0:1].clone()
            injected = teacher.clone()
            injected[:, :, 0:1] = sample[:, :, 0:1]
            counts["teacher_future_injection_steps"] += 1
            return injected

        _patch(model, "_denoise_video", skip_video_dit, patches)
        _patch(
            model.infer_video_scheduler,
            "step",
            install_teacher_future,
            patches,
        )
    else:
        original_video = model._denoise_video
        original_video_step = model.infer_video_scheduler.step

        def count_video_dit(*args, **kwargs):
            counts["video_denoise_loop_calls"] += 1
            counts["video_dit_calls"] += 1
            return original_video(*args, **kwargs)

        def count_video_step(*args, **kwargs):
            counts["video_scheduler_steps"] += 1
            return original_video_step(*args, **kwargs)

        _patch(model, "_denoise_video", count_video_dit, patches)
        _patch(model.infer_video_scheduler, "step", count_video_step, patches)

    original_action = model._denoise_action_with_video_cache

    def count_action_dit(*args, **kwargs):
        if diagnostic["action_initial_noise"] is None:
            initial_noise = kwargs.get("latents_action", args[0] if args else None)
            if initial_noise is None:
                raise RuntimeError("Could not capture the initial action noise")
            if tuple(int(value) for value in initial_noise.shape) != (1, ACTION_HORIZON, 7):
                raise RuntimeError(
                    f"Expected initial action noise [1,{ACTION_HORIZON},7], "
                    f"got {tuple(initial_noise.shape)}"
                )
            diagnostic["action_initial_noise"] = initial_noise.detach().to(
                device="cpu", dtype=runtime.torch.float32
            ).clone()
        counts["action_dit_calls"] += 1
        return original_action(*args, **kwargs)

    _patch(model, "_denoise_action_with_video_cache", count_action_dit, patches)

    if replacing:
        original_prepare = model.video_expert.prepare

        def capture_action_condition(*args, **kwargs):
            result = original_prepare(*args, **kwargs)
            if "x" in kwargs:
                conditioning = kwargs["x"]
            elif args:
                conditioning = args[0]
            else:
                raise RuntimeError("Could not capture the action-conditioning video latents")
            diagnostic["action_conditioning_latents"] = conditioning.detach().clone()
            return result

        _patch(model.video_expert, "prepare", capture_action_condition, patches)

    try:
        with runtime.torch.inference_mode():
            output = model.infer_action(**_infer_kwargs(runtime, datum, seed))
    finally:
        _restore(patches)

    if counts["video_scheduler_steps"] != INFERENCE_STEPS:
        raise RuntimeError(
            f"Expected {INFERENCE_STEPS} video scheduler steps, got {counts['video_scheduler_steps']}"
        )
    if counts["video_denoise_loop_calls"] != INFERENCE_STEPS:
        raise RuntimeError(
            f"Expected {INFERENCE_STEPS} video loop calls, got {counts['video_denoise_loop_calls']}"
        )
    if counts["action_dit_calls"] != INFERENCE_STEPS:
        raise RuntimeError(
            f"Expected {INFERENCE_STEPS} action DiT calls, got {counts['action_dit_calls']}"
        )

    conditioning = diagnostic["action_conditioning_latents"]
    if diagnostic["action_initial_noise"] is None:
        raise RuntimeError("Inference did not expose initial action noise")
    if replacing and conditioning is None:
        raise RuntimeError("Inference did not expose action-conditioning latents")

    metadata = {
        "mode": "teacher_future" if replacing else "native_capture",
        "inference_args": {
            "action_horizon": ACTION_HORIZON,
            "num_video_frames": VIDEO_FRAMES,
            "num_inference_steps": INFERENCE_STEPS,
            "sigma_shift": SIGMA_SHIFT,
            "rand_device": "cpu",
            "compile_action_infer": False,
            "action_infer_mode": "idm",
        },
        **counts,
        "action_initial_noise": diagnostic["action_initial_noise"],
    }
    if replacing:
        result_latents = output["video_latents"]
        future_unchanged = runtime.torch.equal(result_latents[:, :, 1:], teacher[:, :, 1:])
        first_frame_clamped = runtime.torch.equal(
            result_latents[:, :, 0:1], first_frame["value"]
        )
        conditioning_on_device = conditioning.to(device=teacher.device, dtype=teacher.dtype)
        conditioning_future_matches_teacher = runtime.torch.equal(
            conditioning_on_device[:, :, 1:], teacher[:, :, 1:]
        )
        conditioning_first_frame_matches_observation = runtime.torch.equal(
            conditioning_on_device[:, :, 0:1], first_frame["value"]
        )
        teacher_first_frame_matches_observation = runtime.torch.equal(
            teacher[:, :, 0:1], first_frame["value"]
        )
        metadata.update({
            "teacher_future_injection_verified": bool(
                future_unchanged
                and first_frame_clamped
                and counts["teacher_future_injection_steps"] == INFERENCE_STEPS
                and conditioning_future_matches_teacher
                and conditioning_first_frame_matches_observation
                and teacher_first_frame_matches_observation
            ),
            "teacher_future_latents_unchanged": bool(future_unchanged),
            "first_frame_clamp_verified": bool(first_frame_clamped),
            "action_conditioning_latents": conditioning.detach().to(
                device="cpu", dtype=runtime.torch.float32
            ).clone(),
            "action_conditioning_future_matches_teacher": bool(
                conditioning_future_matches_teacher
            ),
            "action_conditioning_first_frame_matches_observation": bool(
                conditioning_first_frame_matches_observation
            ),
            "teacher_first_frame_matches_observation": bool(
                teacher_first_frame_matches_observation
            ),
        })
        if not (
            metadata["teacher_future_injection_verified"]
            and metadata["action_conditioning_future_matches_teacher"]
            and metadata["action_conditioning_first_frame_matches_observation"]
            and metadata["teacher_first_frame_matches_observation"]
        ):
            raise RuntimeError(f"Teacher-future injection verification failed: {metadata}")
    output = dict(output)
    output["_diagnostic"] = {**output.get("_diagnostic", {}), **metadata}
    return output


def run_capture(runtime, datum, seed) -> dict:
    """Run the unmodified native IDM call and count its video/action DiT calls."""
    return _run(runtime, datum, seed)


def run_teacher_future(runtime, datum, seed, teacher_latents) -> dict:
    """Run IDM with teacher future latents injected at every video scheduler step."""
    if teacher_latents is None:
        raise ValueError("teacher_latents is required")
    return _run(runtime, datum, seed, teacher_latents=teacher_latents)
