import copy
import importlib.util
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("aqlm_plus_control_tested", HERE / "control.py")
control = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(control)


def bound_protocol():
    value = copy.deepcopy(control.PROTOCOL)
    job = "26009999.pbs101"
    artifacts = value["sources"]["parent_root"] + "/artifacts/" + job
    value["sources"]["parent_pair_job"] = job
    value["sources"]["parent_pair_artifacts"] = artifacts
    value["sources"]["parent_protocol_file"] = artifacts + "/protocol.json"
    for arm in control.ARMS:
        value["arm_sources"][arm]["source_job"] = job
        value["arm_sources"][arm]["source_protocol_file"] = artifacts + "/protocol.json"
    return value


def arm_row(arm):
    return {
        "protocol": control.PROTOCOL_NAME, "arm": arm, "status": "complete",
        "complete": True, "updates": 8, "training_exposures": 80,
        "calibration": 80, "heldout": 80, "final_anchor": True,
        "native": False, "new_training_updates": 0,
    }


def complete_value():
    attempt = "a"
    name = control._stage_name(attempt)
    job = "26009999.pbs101"
    summary = {
        "protocol": control.PROTOCOL_NAME, "status": "complete", "complete": True,
        "pbs_jobid": job, "episodes": 560,
        "episodes_per_arm": {"reconstruction": 280, "ours": 280},
        "native_low_bit_kernel": False, "new_training_updates": 0,
        "new_teacher_queries": 0, "reference_episode_slots": 0,
        "overall": {"arms": {arm: {"episodes": 280} for arm in control.ARMS},
                    "ours_vs_reconstruction": {"episodes": 280}},
    }
    workers = {}
    for worker in range(4):
        workers[str(worker)] = {"rc": "0", "summary": {
            "protocol": control.PROTOCOL_NAME, "worker_id": worker,
            "arm": "reconstruction" if worker < 2 else "ours",
            "status": "complete", "complete": True, "episodes": 140,
        }}
    return {
        "handle": {"attempt": attempt, "job_id": job, "job_name": name},
        "scheduler": {"job_id": job, "query_rc": 0, "job_state": "F",
                      "Exit_status": "0", "Job_Name": name, "queue": "g3"},
        "artifact": {"exit_code": "0", "PHASE_COMPLETE": True,
                     "PIPELINE_COMPLETE": True, "PIPELINE_CHECKPOINTED": False},
        "results": {"summary": summary, "workers": workers},
        "protocol_matches": True,
    }


class ControllerTests(unittest.TestCase):
    def test_protocol_and_resources_are_single_4gpu_16h_job(self):
        self.assertEqual(control.RESOURCES["gpus"], 4)
        self.assertEqual(control.RESOURCES["walltime"], "16:00:00")
        self.assertEqual(control.PROTOCOL["boundary"]["closed_loop_authorized"], True)
        self.assertEqual(control.PROTOCOL["allocation"]["gpu_monitor_seconds"], 15)
        self.assertEqual(control.PROTOCOL["worker_assignment"]["variants_per_worker"], 140)

    def test_qsub_has_pair_stage_and_exact_resources(self):
        command = control._qsub_command("a")
        self.assertIn("-N aqlmplus261026a", command)
        self.assertIn("-l select=1:ncpus=32:mem=220gb:ngpus=4", command)
        self.assertIn("-l walltime=16:00:00", command)
        self.assertIn("-v FW_AQLM_STAGE=pair", command)
        self.assertNotIn("depend=afterok", command)

    def test_unbound_source_is_not_uploadable(self):
        value=copy.deepcopy(control.PROTOCOL)
        value['sources']['parent_pair_job']=None
        value['sources']['parent_pair_artifacts']=None
        self.assertFalse(control._bound_source(value))

    def test_bound_source_requires_job_and_artifact_identity(self):
        value = bound_protocol()
        self.assertTrue(control._bound_source(value))
        value["arm_sources"]["ours"]["source_protocol_file"] = value["sources"]["parent_protocol_file"] + ".other"
        self.assertFalse(control._bound_source(value))

    def test_completion_gate_requires_exact_560_and_each_arm(self):
        self.assertTrue(control.complete_gate(complete_value()))
        value = complete_value()
        value["results"]["summary"]["episodes"] = 559
        self.assertFalse(control.complete_gate(value))
        value = complete_value()
        value["results"]["summary"]["overall"]["arms"]["ours"]["episodes"] = 279
        self.assertFalse(control.complete_gate(value))
        value = complete_value()
        value["scheduler"]["job_state"] = "R"
        self.assertFalse(control.complete_gate(value))

    def test_completion_gate_rejects_native_or_new_training(self):
        value = complete_value()
        value["results"]["summary"]["native_low_bit_kernel"] = True
        self.assertFalse(control.complete_gate(value))
        value = complete_value()
        value["results"]["summary"]["new_training_updates"] = 1
        self.assertFalse(control.complete_gate(value))

    def test_parent_source_gate_requires_fixed_parent_evidence(self):
        parent = {
            "completion_passed": True,
            "stages": {"pair": {
                "job_id": "26008888.pbs101",
                "scheduler": {"query_rc": 0, "job_state": "F", "Exit_status": "0"},
                "artifact": {"exit_code": "0", "phase_complete": True},
                "summary": {"complete": True, "status": "complete",
                            "closed_loop_episodes": 0, "episodes": 0},
                "arms": {arm: {"summary": {
                    "arm": arm, "complete": True, "updates": 8,
                    "exposures": 80, "calibration": 80, "heldout": 80,
                    "final_anchor": True,
                }} for arm in control.ARMS},
            }},
        }
        job, artifacts = control._source_gate(parent)
        self.assertEqual(job, "26008888.pbs101")
        self.assertTrue(artifacts.endswith("/artifacts/26008888.pbs101"))

    def test_parent_source_gate_rejects_missing_anchor(self):
        parent = {
            "completion_passed": True,
            "stages": {"pair": {
                "job_id": "26008888.pbs101",
                "scheduler": {"query_rc": 0, "job_state": "F", "Exit_status": "0"},
                "artifact": {"exit_code": "0", "phase_complete": True},
                "summary": {"complete": True, "status": "complete", "episodes": 0},
                "arms": {arm: {"summary": {
                    "arm": arm, "complete": True, "updates": 8,
                    "exposures": 80, "calibration": 80, "heldout": 80,
                    "final_anchor": arm == "reconstruction",
                }} for arm in control.ARMS},
            }},
        }
        with self.assertRaises(RuntimeError):
            control._source_gate(parent)

    def test_resume_only_accepts_checkpoint_or_receipted_failure(self):
        scheduler = {"query_rc": 0, "job_state": "F", "Exit_status": "143",
                     "job_id": "26009999.pbs101"}
        failure = {"protocol": control.PROTOCOL_NAME,
                   "pbs_jobid": "26009999.pbs101", "reason": "worker stopped"}
        self.assertTrue(control._is_explicit_failure_resume(scheduler, {"failure": failure}, None))
        self.assertFalse(control._is_explicit_failure_resume(
            scheduler, {"failure": None}, None))
        self.assertFalse(control._is_explicit_failure_resume(
            {**scheduler, "Exit_status": "124"}, {"failure": None}, None))

    def test_pbs_and_source_files_are_small_and_guarded(self):
        for name in ("control.py", "run.pbs", "environment.sh", "protocol.json"):
            self.assertLessEqual((HERE / name).stat().st_size, 256 * 1024)
        run = (HERE / "run.pbs").read_text(encoding="utf-8")
        env = (HERE / "environment.sh").read_text(encoding="utf-8")
        self.assertIn("qstat -xf", run)
        self.assertIn("exec_host", run)
        self.assertIn("sleep 15", run)
        self.assertIn('evaluate.py', run)
        self.assertIn("reference_manifest", run)
        self.assertIn('CUDA_VISIBLE_DEVICES="${GPU_UUIDS[0]}" run_python', run)
        self.assertIn("FW_JOB_START_EPOCH", env)
        self.assertIn("$AQLM_VENDOR", env)
        self.assertNotIn("$NEW_ROOT/vendor/AQLM", env)

    def test_uploaded_contract_contains_cl_entrypoints(self):
        self.assertTrue({"runtime.py", "evaluate.py", "aggregate.py"}.issubset(control.FILES))


if __name__ == "__main__":
    unittest.main()

