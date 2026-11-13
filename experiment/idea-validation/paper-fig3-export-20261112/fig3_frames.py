"""Decoded BF16 futures for Fig. 3(b): the camera images and the predicted frames of a few calibration contexts.

  fig3_frames.py <laq protocol.json> <out dir> <count>

Takes `count` contexts spread evenly over the C80 cohort. For each one it runs the BF16 query at sampler seed index 0
(the query of the reader-metric teacher), decodes the video latents with the model's VAE and writes
  <out>/<case>/frame0.png .. frame8.png    448 x 224, third-person view left, wrist view right; frame 0 is the
                                            observed frame, frames 1-4 decode the first predicted latent frame
  <out>/<case>/agentview_image.png, robot0_eye_in_hand_image.png   the two cameras as the policy sees them
  <out>/index.json                          case, suite, task, latent shape, agreement with the cached LAQ teacher
"""
import json
from pathlib import Path
import sys


def main():
    protocol, out, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    import numpy as np
    import torch
    from PIL import Image
    import calibration_source
    import fw_common
    from profile_worker import _capture_anchor

    fw_common.guard()
    p = fw_common.read(protocol)
    m = p["models"]["fastwam"]
    plan_path = Path(m["panel"]["calibration_plan"])
    plan = fw_common.read(plan_path)
    rows = {r["case_id"]: r for r in plan["inputs"]}
    cohort = plan["calibration_cohorts"]["c80"]
    cases = [cohort[round(k * (len(cohort) - 1) / (count - 1))] for k in range(count)]
    teachers = Path(p["remote_root"]) / "results" / "fastwam" / "calib" / "teacher"

    runtime = fw_common.bf16_runtime(out / "runtime", text_encoder=True)
    model = runtime.model
    index = []
    for case in cases:
        observation, metadata = calibration_source.load_observation(np, plan_path.parent, rows[case])
        runtime.description = metadata["description"]
        seed = rows[case]["sampler_seeds"][0]
        action, _, future, calls = _capture_anchor(runtime, runtime.datum(observation), seed, torch)
        if calls != 1:
            raise RuntimeError("A query called infer_action more than once")
        with torch.inference_mode():
            frames = model._decode_latents(future.to(model.device), tiled=False)
        folder = out / str(case)
        folder.mkdir(parents=True, exist_ok=True)
        for t, frame in enumerate(frames):
            frame.save(folder / f"frame{t}.png")
        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            Image.fromarray(np.ascontiguousarray(observation[key][::-1, ::-1])).save(folder / f"{key}.png")
        record = dict(case=case, suite=rows[case]["suite"], task=metadata["description"], sampler_seed=seed,
                      latent_shape=list(future.shape), frames=len(frames), size=list(frames[0].size))
        cached = teachers / f"{case}.pt"
        if cached.exists():
            reference = torch.load(cached, map_location="cpu", weights_only=True)["future"]
            record["teacher_future_max_abs"] = float((future.float().cpu() - reference.float()).abs().max())
        index.append(record)
        fw_common.save(out / "index.json", dict(protocol=p["protocol"], cases=index))
        print("FRAMES", json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
