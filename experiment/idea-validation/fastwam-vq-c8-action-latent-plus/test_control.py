"""Offline unit tests for the durable retry/resume controller."""
from contextlib import ExitStack, nullcontext
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location("c8_control_test_target", Path(__file__).with_name("control.py"))
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


class RestartTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.files = {
            control.CONTROL_ROOT + "/latest_handle.json": {
                "job_id": "25814264.pbs101", "status": "confirmed", "resume_sequence": 0,
            }
        }
        self.events = []
        self.stack.enter_context(patch.object(
            control.remote, "sftp_client", side_effect=lambda _transport: nullcontext(object())))
        self.stack.enter_context(patch.object(
            control, "_remote_json", side_effect=lambda _sftp, path: self.files.get(path)))
        self.stack.enter_context(patch.object(
            control, "_save_remote_json",
            side_effect=lambda _sftp, path, value: self.files.__setitem__(path, dict(value))))
        self.command = Mock(side_effect=self._command)
        self.stack.enter_context(patch.object(control.remote, "command", self.command))
        self.matches = self.stack.enter_context(patch.object(control, "_query_matches", return_value=[]))
        self.scheduler = self.stack.enter_context(patch.object(control, "scheduler"))
        self.require_uploaded = self.stack.enter_context(patch.object(
            control, "_require_uploaded", side_effect=lambda _transport: self.events.append("uploaded")))

    def _command(self, _transport, command, **_kwargs):
        self.events.append("qsub")
        return 0, "25814265.pbs101"

    def _failed(self):
        return {
            "query_rc": 0, "job_state": "F", "Exit_status": "1",
            "exit_code.txt": "1", "PIPELINE_COMPLETE": False,
        }

    def test_running_unknown_and_completed_jobs_cannot_be_retried(self):
        cases = {
            "running": {"query_rc": 0, "job_state": "R"},
            "unknown": {"query_rc": 1, "job_state": None},
            "successful_complete": {
                "query_rc": 0, "job_state": "F", "Exit_status": "0",
                "exit_code.txt": "0", "PIPELINE_COMPLETE": True,
            },
        }
        for name, status in cases.items():
            with self.subTest(name=name):
                self.scheduler.return_value = status
                with self.assertRaises(RuntimeError):
                    control.retry(None)
                self.command.assert_not_called()
                self.require_uploaded.assert_not_called()

    def test_failed_job_retries_with_metadata_after_protocol_check(self):
        self.scheduler.return_value = self._failed()

        result = control.retry(None)

        self.assertEqual(result["job_name"], "fwalp1015r1")
        self.assertEqual(result["retry"], True)
        self.assertEqual(result["resumes_job_id"], "25814264.pbs101")
        self.assertEqual(result["job_id"], "25814265.pbs101")
        self.assertEqual(self.events, ["uploaded", "qsub"])
        command = self.command.call_args.args[1]
        self.assertIn("/artifacts/fwalp1015r1.pbs.out", command)

    def test_unresolved_intent_blocks_another_submission(self):
        self.files[control.CONTROL_ROOT + "/resume_intent_1.json"] = {
            "status": "unknown_or_rejected", "retry": True,
        }
        self.scheduler.return_value = self._failed()

        with self.assertRaisesRegex(RuntimeError, "no second qsub"):
            control.retry(None)

        self.command.assert_not_called()
        self.scheduler.assert_not_called()

    def test_matching_retry_job_recovers_handle_without_qsub(self):
        job_id = "25814300.pbs101"
        self.files[control.CONTROL_ROOT + "/resume_intent_1.json"] = {
            "status": "intent_before_qsub", "retry": True,
            "resumes_job_id": "25814264.pbs101",
        }
        self.matches.return_value = [job_id]

        result = control.retry(None)

        self.assertEqual(result["job_id"], job_id)
        self.assertEqual(result["resumes_job_id"], "25814264.pbs101")
        self.assertTrue(result["retry"])
        self.assertEqual(self.files[control.CONTROL_ROOT + "/latest_handle.json"]["job_id"], job_id)
        self.command.assert_not_called()
        self.scheduler.assert_not_called()
        self.require_uploaded.assert_not_called()

    def test_resume_still_rejects_a_failed_pipeline(self):
        self.scheduler.return_value = self._failed()

        with self.assertRaisesRegex(RuntimeError, "checkpointed pipeline"):
            control.resume(None)

        self.command.assert_not_called()
        self.require_uploaded.assert_not_called()


class LaneTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.files = {}
        self.sftp = Mock()
        self.sftp.stat.return_value = Mock(st_size=100)
        self.stack.enter_context(patch.object(
            control.remote, "sftp_client", side_effect=lambda _transport: nullcontext(self.sftp)))
        self.stack.enter_context(patch.object(
            control, "_remote_json", side_effect=lambda _sftp, path: self.files.get(path)))
        self.stack.enter_context(patch.object(
            control, "_save_remote_json",
            side_effect=lambda _sftp, path, value: self.files.__setitem__(path, dict(value))))
        self.matches = self.stack.enter_context(patch.object(control, "_query_matches", return_value=[]))
        self.command = Mock(return_value=(0, "25830001.pbs101"))
        self.stack.enter_context(patch.object(control.remote, "command", self.command))
        self.uploaded = self.stack.enter_context(patch.object(control, "_require_uploaded"))

    def test_train_submits_use_separate_lanes_and_do_not_update_generic_latest(self):
        self.command.side_effect = [(0, "25830001.pbs101"), (0, "25830002.pbs101")]

        first = control.train_submit(None, "rho0.1")
        second = control.train_submit(None, "rho1.5")

        first_intent = self.files[control.CONTROL_ROOT + "/train_rho0.1_intent.json"]
        second_intent = self.files[control.CONTROL_ROOT + "/train_rho1.5_intent.json"]
        self.assertEqual(first["arm"], "rho0.1")
        self.assertEqual(second["arm"], "rho1.5")
        self.assertEqual(first_intent["job_name"], "fwalt15r01")
        self.assertEqual(second_intent["job_name"], "fwalt15r15")
        self.assertEqual(first_intent["script"], control.TRAINING_SCRIPT)
        self.assertEqual(first_intent["stage"], "training")
        self.assertEqual(first_intent["select"], "1:ncpus=16:mem=110gb:ngpus=1")
        self.assertEqual(second_intent["walltime"], "01:59:00")
        self.assertEqual(first_intent["env"], {"FW_TRAIN_ARM": "rho0.1"})
        self.assertNotIn(control.CONTROL_ROOT + "/latest_handle.json", self.files)
        self.assertIn("-v FW_TRAIN_ARM=rho0.1", self.command.call_args_list[0].args[1])
        self.assertIn("train_one.pbs", self.command.call_args_list[0].args[1])
        self.assertEqual(self.uploaded.call_count, 2)

    def test_existing_running_handle_is_returned_without_another_qsub(self):
        arm = "rho0.1"
        intent_path, handle_path = control._lane_record_paths("training", arm)
        handle = {"job_id": "25830001.pbs101", "job_name": "fwalt15r01",
                  "stage": "training", "arm": arm, "status": "confirmed"}
        self.files[intent_path] = dict(handle)
        self.files[handle_path] = dict(handle)
        scheduler = self.stack.enter_context(patch.object(control, "scheduler", return_value={
            "query_rc": 0, "job_state": "Q", "Job_Name": "fwalt15r01",
        }))

        result = control.train_submit(None, arm)

        self.assertEqual(result["handle"]["job_id"], handle["job_id"])
        self.assertEqual(result["scheduler"]["job_state"], "Q")
        scheduler.assert_called_once()
        self.command.assert_not_called()
        self.uploaded.assert_not_called()

    def test_unresolved_train_intent_without_matching_job_blocks_second_qsub(self):
        path = control.CONTROL_ROOT + "/train_rho1.5_intent.json"
        self.files[path] = {"status": "unknown_or_rejected", "stage": "training", "arm": "rho1.5"}

        with self.assertRaisesRegex(RuntimeError, "no second qsub"):
            control.train_submit(None, "rho1.5")

        self.command.assert_not_called()
        self.uploaded.assert_not_called()

    def test_qselect_match_recovers_unknown_train_intent(self):
        arm = "rho1.5"
        intent_path, handle_path = control._lane_record_paths("training", arm)
        self.files[intent_path] = {"status": "intent_before_qsub", "stage": "training", "arm": arm}
        self.matches.return_value = ["25830002.pbs101"]
        self.stack.enter_context(patch.object(control, "scheduler", return_value={
            "query_rc": 0, "job_state": "R", "Job_Name": "fwalt15r15",
        }))

        result = control.train_submit(None, arm)

        self.assertEqual(result["handle"]["job_id"], "25830002.pbs101")
        self.assertEqual(self.files[handle_path]["status"], "confirmed")
        self.command.assert_not_called()
        self.uploaded.assert_not_called()

    def _completed_training_status(self, arm, job_id):
        config = control.PROTOCOL["arm_configs"][arm]
        case_ids = control.PROTOCOL["calibration"]["case_ids"]
        source = {"protocol": "fastwam-vq-calibration-suites-v1", "cohort": "c8",
                  "source_job_id": control.PROTOCOL["calibration"]["source_job_id"],
                  "case_ids": case_ids}
        summary = {"schema_version": 1, "protocol": control.PROTOCOL["protocol"],
                   "pbs_jobid": job_id, "job_name": control.TRAINING_JOB_NAMES[arm],
                   "arm": arm, "rho": config["rho"], "status": "complete", "complete": True,
                   "updates": config["updates"], "epochs_completed": config["epochs"],
                   "source_job_id": source["source_job_id"]}
        receipt = {"schema_version": 1, "protocol": control.PROTOCOL["protocol"],
                   "status": "complete", "complete": True, "arm": arm,
                   "rho": config["rho"],
                   "epochs": config["epochs"], "epochs_completed": config["epochs"],
                   "updates": config["updates"], "target_linears": 614,
                   "case_ids": case_ids, "source_c8": source, "normalization_c": 0.25,
                   "optimizer_reset_after_update1": True,
                   "counts": {"gradient_updates": config["updates"], "reference_contexts": 8}}
        return {"query_rc": 0, "job_state": "F", "Job_Name": control.TRAINING_JOB_NAMES[arm],
                "Exit_status": "0", "exit_code.txt": "0", "TRAINING_COMPLETE": True,
                "TRAINING_CHECKPOINTED": False, "training_summary.json": summary,
                "training_receipt.json": receipt}

    def _training_handles(self):
        jobs = {"rho0.1": "25830001.pbs101", "rho1.5": "25830002.pbs101"}
        for arm, job_id in jobs.items():
            intent_path, handle_path = control._lane_record_paths("training", arm)
            record = {"job_id": job_id, "job_name": control.TRAINING_JOB_NAMES[arm],
                      "script": control.TRAINING_SCRIPT, "stage": "training", "arm": arm,
                      "status": "confirmed"}
            record.update({key: control.TRAINING_RESOURCES[key]
                           for key in ("queue", "select", "gpus", "ncpus", "memory_gb", "walltime")})
            self.files[intent_path] = dict(record)
            self.files[handle_path] = dict(record)
        return jobs

    def _failed_complete_validation_status(self, arm, job_id):
        status = self._completed_training_status(arm, job_id)
        status.update({"Exit_status": "1", "exit_code.txt": "1",
                       "TRAINING_COMPLETE": False, "TRAINING_CHECKPOINTED": False})
        summary = status["training_summary.json"]
        summary.update({"status": "failed", "complete": False,
                        "failed_phase": "complete_validation"})
        return status

    def _install_cpu_validation_handle(self, arm, source_job_id, validation_job_id):
        train_intent_path, train_handle_path = control._lane_record_paths("training", arm)
        gpu_handle = dict(self.files[train_handle_path])
        self.files[control.CONTROL_ROOT + f"/train_{arm}_gpu_handle.json"] = gpu_handle
        self.files[control.CONTROL_ROOT + f"/train_{arm}_gpu_intent.json"] = dict(
            self.files[train_intent_path])
        self.assertEqual(gpu_handle["job_id"], source_job_id)
        handle = {"job_id": validation_job_id, "job_name": control.VALIDATION_JOB_NAMES[arm],
                  "script": control.VALIDATION_SCRIPT, "stage": "validation", "arm": arm,
                  "status": "confirmed", "purpose": "validation_only",
                  "source_training_job_id": source_job_id, **control.VALIDATION_RESOURCES}
        validate_intent_path, validate_handle_path = control._lane_record_paths("validation", arm)
        self.files[validate_intent_path] = dict(handle)
        self.files[validate_handle_path] = dict(handle)
        self.files[train_intent_path] = dict(handle)
        self.files[train_handle_path] = dict(handle)
        return handle

    def _cpu_validation_status(self, arm, job_id, source_job_id):
        status = self._completed_training_status(arm, job_id)
        status["Job_Name"] = control.VALIDATION_JOB_NAMES[arm]
        status["training_summary.json"].update({
            "job_name": control.VALIDATION_JOB_NAMES[arm],
            "phase": "metadata_validation", "purpose": "validation_only",
            "source_training_job_id": source_job_id, "new_model_forwards": 0,
        })
        return status

    def test_evaluation_checks_both_receipts_and_ignores_cancelled_legacy_latest(self):
        jobs = self._training_handles()
        self.files[control.CONTROL_ROOT + "/latest_handle.json"] = {
            "job_id": "25821241.pbs101", "job_name": control.JOB_NAME, "status": "confirmed",
        }
        statuses = {job_id: self._completed_training_status(arm, job_id)
                    for arm, job_id in jobs.items()}
        scheduler = self.stack.enter_context(patch.object(
            control, "scheduler", side_effect=lambda _transport, job_id, **_kwargs: statuses[job_id]))
        stat_files = self.stack.enter_context(patch.object(control, "_remote_file_exists", return_value=True))
        self.command.return_value = (0, "25830003.pbs101")

        result = control.evaluation_submit(None)

        eval_intent = self.files[control.CONTROL_ROOT + "/eval_intent.json"]
        self.assertEqual(result["stage"], "evaluation")
        self.assertEqual(result["job_name"], "fwale1015")
        self.assertEqual(eval_intent["restart_job_prefix"], "fwale1015")
        self.assertEqual(eval_intent["script"], control.EVALUATION_SCRIPT)
        self.assertEqual(self.files[control.CONTROL_ROOT + "/latest_handle.json"]["job_id"],
                         "25830003.pbs101")
        self.assertEqual(scheduler.call_count, 2)
        self.assertEqual(stat_files.call_count, 4)
        self.assertIn("fwale1015", self.command.call_args.args[1])
        self.assertNotIn("25821241", self.command.call_args.args[1])

    def test_evaluation_refuses_checkpointed_arm_before_qsub(self):
        jobs = self._training_handles()
        statuses = {job_id: self._completed_training_status(arm, job_id)
                    for arm, job_id in jobs.items()}
        statuses[jobs["rho1.5"]]["TRAINING_COMPLETE"] = False
        statuses[jobs["rho1.5"]]["TRAINING_CHECKPOINTED"] = True
        self.stack.enter_context(patch.object(
            control, "scheduler", side_effect=lambda _transport, job_id, **_kwargs: statuses[job_id]))
        self.stack.enter_context(patch.object(control, "_remote_file_exists", return_value=True))

        with self.assertRaisesRegex(RuntimeError, "not authoritatively complete"):
            control.evaluation_submit(None)

        self.command.assert_not_called()
        self.assertNotIn(control.CONTROL_ROOT + "/eval_handle.json", self.files)

    def test_training_validate_submits_only_for_complete_validation_failure_and_promotes_cpu_role(self):
        arm = "rho1.5"
        source_job_id = "25830002.pbs101"
        jobs = self._training_handles()
        self.assertEqual(jobs[arm], source_job_id)
        statuses = {source_job_id: self._failed_complete_validation_status(arm, source_job_id)}
        self.stack.enter_context(patch.object(
            control, "scheduler", side_effect=lambda _transport, job_id, **_kwargs: statuses[job_id]))
        self.stack.enter_context(patch.object(control, "_remote_file_exists", return_value=True))
        self.command.return_value = (0, "25830010.pbs101")

        result = control.training_validate(None, arm)

        validation_intent = self.files[control.CONTROL_ROOT + f"/validate_{arm}_intent.json"]
        canonical = self.files[control.CONTROL_ROOT + f"/train_{arm}_handle.json"]
        archived = self.files[control.CONTROL_ROOT + f"/train_{arm}_gpu_handle.json"]
        self.assertEqual(result["purpose"], "validation_only")
        self.assertEqual(result["source_training_job_id"], source_job_id)
        self.assertEqual(validation_intent["stage"], "validation")
        self.assertEqual(validation_intent["script"], control.VALIDATION_SCRIPT)
        self.assertEqual(validation_intent["queue"], "qdev")
        self.assertEqual(validation_intent["select"], "1:ncpus=1:mem=4gb:ngpus=0")
        self.assertEqual(validation_intent["gpus"], 0)
        self.assertEqual(validation_intent["purpose"], "validation_only")
        self.assertEqual(canonical["job_id"], "25830010.pbs101")
        self.assertEqual(canonical["purpose"], "validation_only")
        self.assertEqual(archived["job_id"], source_job_id)
        self.assertNotIn(control.CONTROL_ROOT + "/latest_handle.json", self.files)
        command = self.command.call_args.args[1]
        self.assertIn("-v FW_VALIDATE_ARM=rho1.5,FW_SOURCE_TRAIN_JOB_ID=" + source_job_id, command)
        self.assertIn("validate_training.pbs", command)

    def test_training_validate_refuses_wrong_failure_phase_or_incomplete_receipt(self):
        arm = "rho0.1"
        source_job_id = "25830001.pbs101"
        self._training_handles()
        wrong_phase = self._failed_complete_validation_status(arm, source_job_id)
        wrong_phase["training_summary.json"]["failed_phase"] = "training"
        incomplete_receipt = self._failed_complete_validation_status(arm, source_job_id)
        incomplete_receipt["training_receipt.json"]["updates"] = 7
        statuses = {source_job_id: wrong_phase}
        scheduler = self.stack.enter_context(patch.object(
            control, "scheduler", side_effect=lambda _transport, job_id, **_kwargs: statuses[job_id]))
        self.stack.enter_context(patch.object(control, "_remote_file_exists", return_value=True))

        with self.assertRaisesRegex(RuntimeError, "complete_validation failure"):
            control.training_validate(None, arm)
        statuses[source_job_id] = incomplete_receipt
        with self.assertRaisesRegex(RuntimeError, "exact completed training receipt"):
            control.training_validate(None, arm)

        self.command.assert_not_called()
        self.assertEqual(scheduler.call_count, 2)

    def test_unknown_validation_intent_recovers_matching_job_without_qsub(self):
        arm = "rho1.5"
        source_job_id = "25830002.pbs101"
        self._training_handles()
        intent_path, _ = control._lane_record_paths("validation", arm)
        self.files[intent_path] = {
            "job_name": control.VALIDATION_JOB_NAMES[arm], "stage": "validation",
            "arm": arm, "status": "unknown_or_rejected",
            "source_training_job_id": source_job_id,
        }
        self.matches.return_value = ["25830011.pbs101"]
        self.stack.enter_context(patch.object(control, "scheduler", return_value={
            "query_rc": 0, "job_state": "Q", "Job_Name": control.VALIDATION_JOB_NAMES[arm],
        }))

        result = control.training_validate(None, arm)

        self.assertEqual(result["handle"]["job_id"], "25830011.pbs101")
        self.assertEqual(result["handle"]["purpose"], "validation_only")
        self.assertEqual(self.files[control.CONTROL_ROOT + f"/train_{arm}_handle.json"]["job_id"],
                         "25830011.pbs101")
        self.command.assert_not_called()
        self.uploaded.assert_not_called()

    def test_unresolved_validation_intent_without_match_blocks_another_qsub(self):
        arm = "rho0.1"
        self._training_handles()
        intent_path, _ = control._lane_record_paths("validation", arm)
        self.files[intent_path] = {
            "job_name": control.VALIDATION_JOB_NAMES[arm], "stage": "validation",
            "arm": arm, "status": "unknown_or_rejected",
        }

        with self.assertRaisesRegex(RuntimeError, "no second qsub"):
            control.training_validate(None, arm)

        self.command.assert_not_called()
        self.uploaded.assert_not_called()

    def test_train_status_accepts_an_existing_cpu_validation_role(self):
        arm = "rho1.5"
        source_job_id = "25830002.pbs101"
        self._training_handles()
        handle = self._install_cpu_validation_handle(arm, source_job_id, "25830014.pbs101")
        scheduler = self.stack.enter_context(patch.object(control, "scheduler", return_value={
            "query_rc": 0, "job_state": "Q", "Job_Name": control.VALIDATION_JOB_NAMES[arm],
        }))

        result = control.train_status(None, arm)

        self.assertEqual(result["job_state"], "Q")
        self.assertEqual(result["Job_Name"], control.VALIDATION_JOB_NAMES[arm])
        scheduler.assert_called_once_with(None, handle["job_id"], stage="training", arm=arm)
        self.command.assert_not_called()

    def test_evaluation_accepts_successful_cpu_validation_with_linked_exact_gpu_failure(self):
        jobs = self._training_handles()
        gpu_job_id = jobs["rho1.5"]
        cpu_job_id = "25830012.pbs101"
        self._install_cpu_validation_handle("rho1.5", gpu_job_id, cpu_job_id)
        statuses = {
            jobs["rho0.1"]: self._completed_training_status("rho0.1", jobs["rho0.1"]),
            gpu_job_id: self._failed_complete_validation_status("rho1.5", gpu_job_id),
            cpu_job_id: self._cpu_validation_status("rho1.5", cpu_job_id, gpu_job_id),
        }
        self.stack.enter_context(patch.object(
            control, "scheduler", side_effect=lambda _transport, job_id, **_kwargs: statuses[job_id]))
        self.stack.enter_context(patch.object(control, "_remote_file_exists", return_value=True))
        self.command.return_value = (0, "25830013.pbs101")

        result = control.evaluation_submit(None)

        self.assertEqual(result["job_name"], control.EVALUATION_JOB_NAME)
        self.assertEqual(self.files[control.CONTROL_ROOT + "/eval_handle.json"]["job_id"],
                         "25830013.pbs101")
        self.assertIn("fwale1015", self.command.call_args.args[1])

    def test_evaluation_rejects_cpu_validation_for_other_gpu_failure_phase(self):
        jobs = self._training_handles()
        gpu_job_id = jobs["rho1.5"]
        cpu_job_id = "25830012.pbs101"
        self._install_cpu_validation_handle("rho1.5", gpu_job_id, cpu_job_id)
        failed_source = self._failed_complete_validation_status("rho1.5", gpu_job_id)
        failed_source["training_summary.json"]["failed_phase"] = "training"
        statuses = {
            jobs["rho0.1"]: self._completed_training_status("rho0.1", jobs["rho0.1"]),
            gpu_job_id: failed_source,
            cpu_job_id: self._cpu_validation_status("rho1.5", cpu_job_id, gpu_job_id),
        }
        self.stack.enter_context(patch.object(
            control, "scheduler", side_effect=lambda _transport, job_id, **_kwargs: statuses[job_id]))
        self.stack.enter_context(patch.object(control, "_remote_file_exists", return_value=True))

        with self.assertRaisesRegex(RuntimeError, "complete_validation failure"):
            control.evaluation_submit(None)

        self.command.assert_not_called()
        self.assertNotIn(control.CONTROL_ROOT + "/eval_handle.json", self.files)

if __name__ == "__main__":
    unittest.main()
