#!/usr/bin/env python3
"""Small CPU-PBS validation of the existing PushT dependency overlay."""
import importlib.metadata
import inspect
import json
import os
import pathlib
import sys
import traceback


def package_info(module_name, distribution_name=None):
    module = __import__(module_name)
    try:
        version = importlib.metadata.version(distribution_name or module_name)
    except importlib.metadata.PackageNotFoundError:
        version = getattr(module, "__version__", "unknown")
    return {"version": str(version), "origin": str(inspect.getfile(module))}


def shape_summary(value):
    if isinstance(value, dict):
        return {str(key): shape_summary(item) for key, item in value.items()}
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else type(value).__name__


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: plan_overlay_probe.py REPORT.json")
    report_path = pathlib.Path(sys.argv[1])
    report = {
        "schema": "lpwm-pusht-existing-overlay-probe-v1",
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "execution_host": os.environ.get("HOSTNAME"),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "status": "INCOMPLETE",
    }
    try:
        import numpy
        report["packages"] = {
            "numpy": package_info("numpy"),
            "gym": package_info("gym"),
            "scikit_image": package_info("skimage", "scikit-image"),
            "pymunk": package_info("pymunk"),
            "pygame": package_info("pygame"),
        }
        import torch
        report["packages"]["torch"] = package_info("torch")
        report["torch_cuda_available"] = bool(torch.cuda.is_available())

        import plan  # the pinned official planner entrypoint and registrations

        report["official_plan_import"] = "PASS"
        import gym

        env = gym.make("pusht", with_velocity=True, with_target=True)
        try:
            env.seed(99)
            observation = env.reset()
            report["pusht_env"] = {
                "make": "PASS",
                "reset": "PASS",
                "reset_observation": shape_summary(observation),
                "environment_class": f"{type(env.unwrapped).__module__}.{type(env.unwrapped).__name__}",
            }
        finally:
            env.close()

        report["numpy_runtime_version"] = str(numpy.__version__)
        report["status"] = "RUNTIME_GATE_PASS"
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc(limit=12)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "RUNTIME_GATE_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
