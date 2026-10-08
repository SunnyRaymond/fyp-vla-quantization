"""Minimal control checks for the PBS diagnostic runner."""
import ast
import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("run_diagnostic.py")
SPEC = importlib.util.spec_from_file_location("phase12_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class _FakeQuantizer:
    def __init__(self):
        self.stages = []

    def set_stage(self, stage, step):
        self.stages.append((stage, step))


class _FakeScheduler:
    def build_inference_schedule(self):
        return list(range(10)), list(range(10))

    @staticmethod
    def step(_prediction, delta, sample):
        return sample + delta


class _FakeExpert:
    def prepare(self):
        return "prepared"


class _FakeMot:
    def prefill_video_cache_tensor(self):
        return "prefilled"


class _FakeModel:
    def __init__(self):
        self.infer_video_scheduler = _FakeScheduler()
        self.infer_action_scheduler = _FakeScheduler()
        self.video_expert = _FakeExpert()
        self.mot = _FakeMot()


class DiagnosticRunnerTests(unittest.TestCase):
    def test_each_context_has_two_reusable_endpoints_and_six_mixed_cells(self):
        for case_id in (0, 5, 21):
            for seed_index in (0, 1):
                specs = runner.query_specs(case_id, seed_index)
                self.assertEqual(len(specs), 9)
                self.assertEqual(specs[:3], [
                    ("all_a8", (8, 8, 8), False, False),
                    ("all_a4", (4, 4, 4), False, True),
                    ("independent_reference", (4, 4, 4), True, False),
                ])
                mixed = [item[1] for item in specs[3:]]
                self.assertEqual(set(mixed), set(runner.MIXED_TRIPLES))
                self.assertEqual(len(set(mixed)), 6)

    def test_scheduler_steps_are_tagged_before_each_real_step(self):
        quantizer = _FakeQuantizer()
        model = _FakeModel()
        handles = runner.install_stage_schedulers(model, quantizer)
        conditioning = runner.install_video_conditioning_wrappers(model, quantizer, handles[0][3])
        try:
            for _query in range(2):
                runner.reset_stage_calls(handles)
                video = model.infer_video_scheduler
                timesteps, deltas = video.build_inference_schedule()
                for timestep, delta in zip(timesteps, deltas):
                    video.step(timestep, delta, 0)
                model.video_expert.prepare()
                model.mot.prefill_video_cache_tensor()
                action = model.infer_action_scheduler
                timesteps, deltas = action.build_inference_schedule()
                for timestep, delta in zip(timesteps, deltas):
                    action.step(timestep, delta, 0)
                report = runner.validate_stage_calls(handles)
        finally:
            runner.restore_methods(conditioning)
            runner.restore_stage_schedulers(handles)
        self.assertEqual(report["video"]["observed_steps"], list(range(10)))
        self.assertEqual(report["action"]["observed_steps"], list(range(10)))
        self.assertEqual(quantizer.stages.count(("video_conditioning_prefill", -1)), 4)

    def test_runner_never_calls_episode_task_or_environment_steps(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        forbidden = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in {"task", "reset"}:
                forbidden.append(node.func.attr)
        self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()
