import importlib.util
import json
from contextlib import nullcontext
import errno
from pathlib import Path
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("aqlm_loss_pair_control_tested", HERE / "control.py")
control = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(control)


class _Attrs:
    def __init__(self, size):
        self.st_size = size


class _Stream:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, count):
        return self.data[:count]


class _SFTP:
    def __init__(self, data):
        self.data = data
        self.reads = 0

    def stat(self, _path):
        return _Attrs(len(self.data))

    def file(self, _path, _mode):
        self.reads += 1
        return _Stream(self.data)


class _MemoryStream:
    def __init__(self, fs, path, mode):
        self.fs, self.path, self.mode = fs, path, mode
        self.data = bytearray(fs.get(path, b""))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if "w" in self.mode:
            self.fs[self.path] = bytes(self.data)
        return False

    def read(self, count=-1):
        return bytes(self.data if count < 0 else self.data[:count])

    def write(self, data):
        self.data.extend(data)


class _MemorySFTP:
    def __init__(self, files=None):
        self.files = dict(files or {})

    def stat(self, path):
        if path not in self.files:
            raise OSError(errno.ENOENT, "missing")
        return _Attrs(len(self.files[path]))

    def file(self, path, mode):
        return _MemoryStream(self.files, path, mode)

    def posix_rename(self, source, target):
        self.files[target] = self.files.pop(source)


def _arm(arm):
    return {
        "protocol": control.PROTOCOL_NAME, "arm": arm, "complete": True, "status": "complete",
        "final_update": 8, "calibration": 80, "heldout": 80,
    }


def _complete_receipt():
    attempt = "a"
    job_id = "401234.pbs101"
    job_name = control._stage_name("pair", attempt)
    summaries = {arm: _arm(arm) for arm in control.ARMS}
    return {
        "handle": {"job_id": job_id, "job_name": job_name, "attempt": attempt},
        "scheduler": {"query_rc": 0, "job_state": "F", "Exit_status": "0", "Job_Name": job_name},
        "protocol_matches": True,
        "artifact": {"exit_code": "0", "PHASE_COMPLETE": True},
        "results": {"summary": {"protocol": control.PROTOCOL_NAME, "complete": True,
                                  "status": "complete", "closed_loop_episodes": 0, "arms": summaries},
                    "arms": {arm: {"summary": value} for arm, value in summaries.items()}},
    }


class ControlTests(unittest.TestCase):
    def _fitting_amendment_fixture(self):
        amendment = json.loads((HERE / "fit_budget_amendment.json").read_text(encoding="utf-8"))
        old = json.loads(json.dumps(control.PROTOCOL))
        new = json.loads(json.dumps(control.PROTOCOL))
        old["fitting"] = json.loads(json.dumps(amendment["old_fitting"]))
        new["fitting"] = json.loads(json.dumps(amendment["new_fitting"]))
        return old, new, amendment

    def _minimal_v3_fixture(self):
        current = json.loads(json.dumps(control.PROTOCOL))
        prior = json.loads((HERE / "fit_budget_amendment.json").read_text(encoding="utf-8"))
        old = json.loads(json.dumps(current))
        if old["fitting"].get("profile") == "minimal_v3":
            old["fitting"] = json.loads(json.dumps(prior["old_fitting"]))
        new = json.loads(json.dumps(old))
        new["fitting"].update({
            "init_max_iter": 5, "init_max_points_per_centroid": 16,
            "max_cluster_vectors": 4096, "max_epochs": 1, "steps_per_epoch": 5,
            "relative_mse_tolerance": 0.005, "beam_size": 1,
            "profile": "minimal_v3", "note": "minimal v3 budget",
        })
        amendment = {"protocol": control.PROTOCOL_NAME, "revision": 3,
                     "old_fitting": old["fitting"], "new_fitting": new["fitting"],
                     "user_request": "minimal v3", "allowed_scope": "fitting_only"}
        return old, new, amendment

    def test_protocol_and_resource_contract(self):
        value = control.load_protocol()
        self.assertEqual(value["protocol"], control.PROTOCOL_NAME)
        self.assertEqual(control.RESOURCES["bootstrap"]["queue"], "qdev")
        self.assertEqual(control.RESOURCES["prepare"]["gpus"], 1)
        self.assertEqual(control.RESOURCES["pair"]["gpus"], 2)

    def test_fitting_only_amendment_matches_declared_light_budget(self):
        old, new, amendment = self._fitting_amendment_fixture()
        control._validate_fitting_amendment(old, new, amendment)

    def test_fitting_amendment_rejects_training_change(self):
        old, new, amendment = self._fitting_amendment_fixture()
        new["training"]["updates"] += 1
        with self.assertRaisesRegex(RuntimeError, "outside fitting"):
            control._validate_fitting_amendment(old, new, amendment)

    def test_fitting_amendment_rejects_cohort_change(self):
        old, new, amendment = self._fitting_amendment_fixture()
        new["calibration"]["case_ids"][0] += 1
        with self.assertRaisesRegex(RuntimeError, "outside fitting"):
            control._validate_fitting_amendment(old, new, amendment)

    def test_fitting_amendment_rejects_old_new_mismatch(self):
        old, new, amendment = self._fitting_amendment_fixture()
        amendment["old_fitting"]["max_epochs"] = 99
        with self.assertRaisesRegex(RuntimeError, "old/new"):
            control._validate_fitting_amendment(old, new, amendment)

    def test_minimal_v3_amendment_matches_exact_budget(self):
        old, new, amendment = self._minimal_v3_fixture()
        spec = control._validate_fitting_amendment(old, new, amendment)
        self.assertEqual(spec["archive"], "control/protocol_before_minimal_v3.json")
        self.assertEqual(spec["receipt"], "control/fit_budget_amendment_receipt_v3.json")

    def test_qsub_uses_stage_variable_and_exact_resources(self):
        command = control._qsub_command("prepare", "a")
        self.assertIn("-v FW_AQLM_STAGE=prepare", command)
        self.assertIn("select=1:ncpus=16:mem=180gb:ngpus=1", command)
        self.assertIn("walltime=24:00:00", command)

    def test_complete_gate_requires_both_final8_calib80_heldout80(self):
        self.assertTrue(control.complete_gate(_complete_receipt()))
        value = _complete_receipt()
        value["results"]["arms"]["ours"]["summary"]["heldout"] = 79
        self.assertFalse(control.complete_gate(value))

    def test_complete_gate_rejects_closed_loop(self):
        value = _complete_receipt()
        value["results"]["summary"]["closed_loop_episodes"] = 1
        self.assertFalse(control.complete_gate(value))

    def test_complete_gate_matches_worker_summary_shape_without_status(self):
        value = _complete_receipt()
        for arm in control.ARMS:
            value["results"]["arms"][arm]["summary"].pop("status", None)
        self.assertTrue(control.complete_gate(value))

    def test_explicit_runtime_failure_is_resumable_only_with_failure_receipt(self):
        artifact = {"failure": {"protocol": control.PROTOCOL_NAME,
                                 "pbs_jobid": "401234.pbs101", "error": "runtime failure"}}
        for exit_status in ("1", "124", "143", "271"):
            scheduler = {"query_rc": 0, "job_id": "401234.pbs101", "job_state": "F",
                         "Exit_status": exit_status}
            self.assertTrue(control._is_explicit_failure_resume(scheduler, artifact, None))
        artifact["failure"].pop("error")
        self.assertFalse(control._is_explicit_failure_resume(
            {"query_rc": 0, "job_id": "401234.pbs101", "job_state": "F", "Exit_status": "143"},
            artifact, None))

    def test_exit75_still_requires_checkpoint_receipt(self):
        artifact = {"failure": {"protocol": control.PROTOCOL_NAME,
                                 "pbs_jobid": "401234.pbs101", "reason": "stopped"}}
        scheduler = {"query_rc": 0, "job_id": "401234.pbs101", "job_state": "F", "Exit_status": "75"}
        self.assertFalse(control._is_explicit_failure_resume(scheduler, artifact, None))

    def test_remote_read_is_bounded_before_transfer(self):
        oversized = _SFTP(b"x" * (control.LIMIT + 1))
        with self.assertRaisesRegex(RuntimeError, "256 KiB"):
            control._small_bytes(oversized, "/remote/large.json")
        self.assertEqual(oversized.reads, 0)

    def test_small_remote_json_is_decoded(self):
        value = _SFTP(json.dumps({"phase": "complete"}).encode("utf-8"))
        self.assertEqual(control._json(value, "/remote/phase.json"), {"phase": "complete"})
        self.assertEqual(value.reads, 1)

    def test_stage_compact_surfaces_both_arms(self):
        value = {"stage": "pair", "attempt": "a", "job_id": "1.pbs101",
                 "scheduler": {"job_state": "R", "query_rc": 0},
                 "artifact": {"phase_complete": False, "checkpointed": False},
                 "summary": None, "progress": None,
                 "arms": {arm: {"progress": {"phase": "training", "update": 3,
                                               "calibration": 0, "heldout": 0}, "failure": None}
                          for arm in control.ARMS}}
        compact = control._compact_stage(value)
        self.assertEqual(compact["arms"]["ours"]["update"], 3)
        self.assertEqual(compact["arms"]["reconstruction"]["phase"], "training")

    def test_unresolved_intent_never_reissues_qsub(self):
        state = {"schema_version": 1, "protocol": control.PROTOCOL_NAME, "stage": "bootstrap",
                 "attempts": [{"status": "qsub_outcome_unknown", "job_name": control._stage_name("bootstrap", "a")}]}
        with patch.object(control.remote, "sftp_client", return_value=nullcontext(object())), \
             patch.object(control, "_mkdirs"), patch.object(control, "_state", return_value=state), \
             patch.object(control, "_resolve_latest",
                                                            side_effect=RuntimeError("qsub outcome is unresolved; no second submission will be issued")), \
             patch.object(control, "_named_jobs", return_value=[]), \
             patch.object(control.remote, "command") as command:
            with self.assertRaisesRegex(RuntimeError, "unresolved"):
                control.submit(object(), "bootstrap", "a")
        command.assert_not_called()

    def _transition_sftp(self, *, bootstrap_job="1.pbs101"):
        state = {"schema_version": 1, "protocol": control.PROTOCOL_NAME, "stage": "bootstrap",
                 "attempts": [{"job_id": bootstrap_job, "job_name": control._stage_name("bootstrap", "a"),
                               "attempt": "a", "status": "confirmed"}]}
        return _MemorySFTP({control.CONTROL + "/bootstrap_state.json":
                            (json.dumps(state) + "\n").encode()})

    def test_source_transition_rejects_active_job(self):
        records = {stage: {"attempts": []} for stage in control.STAGES}
        records["bootstrap"]["attempts"] = [{"job_id": "1.pbs101"}]
        with patch.object(control, "_scheduler", return_value={"query_rc": 0, "job_state": "R"}):
            with self.assertRaisesRegex(RuntimeError, "Active PBS job"):
                control._verify_all_stage_schedulers_quiescent(object(), records)

    def test_source_transition_rejects_inconclusive_scheduler(self):
        records = {stage: {"attempts": []} for stage in control.STAGES}
        records["bootstrap"]["attempts"] = [{"job_id": "1.pbs101"}]
        with patch.object(control, "_scheduler", return_value={"query_rc": 1, "job_state": None}):
            with self.assertRaisesRegex(RuntimeError, "inconclusive"):
                control._verify_all_stage_schedulers_quiescent(object(), records)

    def test_unsubmitted_next_stage_transition_requires_verified_parent(self):
        sftp = self._transition_sftp()
        with patch.object(control.remote, "sftp_client", return_value=nullcontext(sftp)), \
             patch.object(control, "_mkdirs"), \
             patch.object(control, "_scheduler", return_value={"query_rc": 0, "job_state": "F", "Exit_status": "0"}), \
             patch.object(control, "_parent_complete", side_effect=RuntimeError("parent is incomplete")):
            with self.assertRaisesRegex(RuntimeError, "parent is incomplete"):
                control.upload(object(), resume_stage="pair")

    def test_unsubmitted_next_stage_transition_is_allowed_after_parent_and_quiescence(self):
        sftp = self._transition_sftp()
        parent = {"job_id": "1.pbs101", "job_name": control._stage_name("prepare", "a")}
        with patch.object(control.remote, "sftp_client", return_value=nullcontext(sftp)), \
             patch.object(control, "_mkdirs"), \
             patch.object(control, "_scheduler", return_value={"query_rc": 0, "job_state": "F", "Exit_status": "0"}), \
             patch.object(control, "_parent_complete", return_value=parent):
            uploaded = control.upload(object(), resume_stage="pair")
        self.assertIn("control.py", uploaded)
        transition = json.loads(sftp.files[control.CONTROL + "/source_transition.json"])
        self.assertEqual(transition["transitions"][-1]["stage"], "pair")
        self.assertEqual(transition["transitions"][-1]["pbs_parent"], parent["job_id"])
        self.assertEqual(json.loads(sftp.files[control.CONTROL + "/bootstrap_state.json"])["attempts"][0]["job_id"],
                         "1.pbs101")

    def test_valid_fitting_amendment_archives_then_publishes_protocol_atomically(self):
        old, new, amendment = self._fitting_amendment_fixture()
        old_raw = (json.dumps(old, indent=2) + "\n").encode()
        new_raw = (json.dumps(new, indent=2) + "\n").encode()
        amendment_raw = (json.dumps(amendment, indent=2) + "\n").encode()
        sftp = _MemorySFTP({control.ROOT + "/protocol.json": old_raw,
                            control.RESULTS + "/prepare/failure.json":
                            (json.dumps({"protocol": control.PROTOCOL_NAME,
                                         "pbs_jobid": "2.pbs101", "reason": "stopped"}) + "\n").encode()})
        records = {"bootstrap": {"attempts": []},
                   "prepare": {"attempts": [{"job_id": "2.pbs101", "attempt": "a",
                                               "job_name": control._stage_name("prepare", "a"),
                                               "status": "confirmed"}]},
                   "pair": {"attempts": []}}
        original_local = control._local_bytes
        def local_bytes(name):
            if name == "protocol.json":
                return new_raw
            if name == "fit_budget_amendment.json":
                return amendment_raw
            return original_local(name)
        original_is_file = Path.is_file
        with patch.object(Path, "is_file", autospec=True,
                          side_effect=lambda path: path.name == "fit_budget_amendment.json"
                          or original_is_file(path)), \
             patch.object(control, "_local_bytes", side_effect=local_bytes), \
             patch.object(control, "PROTOCOL", new), \
             patch.object(control, "_scheduler", return_value={"query_rc": 0, "job_state": "F",
                                                                  "Exit_status": "143", "job_id": "2.pbs101"}), \
             patch.object(control, "_parent_complete", return_value={"job_id": "1.pbs101"}):
            self.assertTrue(control._apply_fitting_amendment(object(), sftp, records, "prepare"))
        self.assertEqual(sftp.files[control.ROOT + "/protocol.json"], new_raw)
        revision = amendment["revision"]
        archive_name = "protocol_before_light_v2.json" if revision == 2 else "protocol_before_minimal_v3.json"
        receipt_name = "fit_budget_amendment_receipt.json" if revision == 2 else "fit_budget_amendment_receipt_v3.json"
        self.assertEqual(sftp.files[control.ROOT + "/control/" + archive_name], old_raw)
        receipt = json.loads(sftp.files[control.ROOT + "/control/" + receipt_name])
        self.assertEqual(receipt["pbs_parent"], "2.pbs101")

    def test_fitting_amendment_rejects_existing_pair_attempt(self):
        old, new, amendment = self._fitting_amendment_fixture()
        old_raw = (json.dumps(old, indent=2) + "\n").encode()
        new_raw = (json.dumps(new, indent=2) + "\n").encode()
        sftp = _MemorySFTP({control.ROOT + "/protocol.json": old_raw})
        records = {"bootstrap": {"attempts": []}, "prepare": {"attempts": []},
                   "pair": {"attempts": [{"job_id": "3.pbs101"}]}}
        with patch.object(Path, "is_file", autospec=True, return_value=True), \
             patch.object(control, "_local_bytes", side_effect=lambda name: new_raw if name == "protocol.json" else
                          (json.dumps(amendment) + "\n").encode()):
            with self.assertRaisesRegex(RuntimeError, "pair submission"):
                control._apply_fitting_amendment(object(), sftp, records, "prepare")


if __name__ == "__main__":
    unittest.main()
