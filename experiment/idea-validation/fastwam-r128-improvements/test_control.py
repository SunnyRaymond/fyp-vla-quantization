import unittest
from contextlib import nullcontext
from unittest.mock import patch

import control


def lane_observation(kind, protocol, job_id="25900001.pbs101"):
    lane = protocol["jobs"][kind]["lanes"][0]
    lane_plan = protocol["lanes"][lane]
    expected_updates = lane_plan.get("updates", lane_plan.get("new_training_updates", 0))
    exposures = lane_plan.get("training_exposures", 0)
    c80 = 80 if lane == "coverage_c80" else 0
    arms = {}
    for spec in lane_plan["arms"]:
        movement = 0.0 if spec.get("t") == 0 else 0.01
        arms[spec["name"]] = {
            "updates": expected_updates,
            "training_exposures": exposures,
            "calibration_queries": 8,
            "heldout_queries": 80,
            "c80_queries": c80,
            "all_parameters_frozen": True,
            "formal_infer_max_abs": 0.0,
            "source_replay_metric_max_abs": 0.0 if spec.get("reused") else None,
            "orthogonality": {"action": 1e-7, "video": 1e-7},
            "movement": {"action": movement, "video": movement},
        }
    summary = {
        "protocol": control.PROTOCOL_ID, "status": "complete", "complete": True,
        "pbs_jobid": job_id, "lane": lane, "target_linears": 614,
        "new_teacher_queries": 0, "arms": arms,
    }
    comparison = {
        "protocol": control.PROTOCOL_ID, "complete": True, "lane": lane, "target_linears": 614,
        "calibration": {spec["name"]: {"queries": 8} for spec in lane_plan["arms"]},
        "heldout": {spec["name"]: {"queries": 80} for spec in lane_plan["arms"]},
    }
    if lane == "coverage_c80":
        comparison["c80"] = {spec["name"]: {"queries": 80} for spec in lane_plan["arms"]}
    if lane == "path":
        training = {"protocol": control.PROTOCOL_ID, "lane": lane, "updates": 0, "history": []}
        checkpoints = {"state": 1024}
    else:
        contexts_per_update = lane_plan.get("contexts_per_update", 1)
        training = {"protocol": control.PROTOCOL_ID, "lane": lane, "arms": {}}
        for name in arms:
            training["arms"][name] = {
                "updates": expected_updates, "training_exposures": exposures,
                "history": [{"update": i, "contexts": [{} for _ in range(contexts_per_update)]}
                            for i in range(1, expected_updates + 1)],
            }
        checkpoints = {name: 1024 for name in arms}
    results = {
        "summary.json": summary,
        "progress.json": {"protocol": control.PROTOCOL_ID, "lane": lane,
                           "status": "complete", "complete": True},
        "selfcheck.json": {"passed": True,
                           "fixtures": [{"initial_H_A4_exact": True, "checkpoint_gradient_exact": True}]},
        "training.json": training,
        "comparison.json": comparison,
    }
    return lane, summary, {
        "lane": lane, "result_json": results,
        "result_files": {name: True for name in control.RESULT_FILES},
        "worker_summary": dict(summary), "artifact_exit_code": 0, "phase_complete": True,
        "checkpoint_size": 1024, "checkpoint_sizes": checkpoints,
        "reuse_receipt": None, "reuse_verified": False,
    }


def passing_observation(kind):
    protocol = control.load_protocol()
    job_id = "25900001.pbs101"
    lane, summary, lane_obs = lane_observation(kind, protocol, job_id)
    observation = {
        "handle": {"job_id": job_id, "job_kind": kind}, "job_kind": kind, "lane": lane,
        "scheduler": {"query_rc": 0, "job_state": "F", "Exit_status": "0",
                      "queue": protocol["jobs"][kind]["execution_queue"]},
        "artifact_exit_code": 0, "phase_complete": True,
        "checkpoint_size": lane_obs["checkpoint_size"], "checkpoint_sizes": lane_obs["checkpoint_sizes"],
        "result_json": lane_obs["result_json"], "result_files": lane_obs["result_files"],
        "worker_summary": dict(summary), "lane_observations": {lane: lane_obs},
    }
    return protocol, observation


class VerificationTests(unittest.TestCase):
    def test_all_frozen_lanes_pass_their_protocol_gates(self):
        for kind in control.KINDS:
            with self.subTest(kind=kind):
                protocol, observation = passing_observation(kind)
                self.assertEqual(control.verification_failures(observation, protocol, kind), [])

    def test_arm_counts_frozen_state_parity_orthogonality_and_teachers_are_gated(self):
        protocol, observation = passing_observation("coverage")
        lane = protocol["jobs"]["coverage"]["lanes"][0]
        arm = next(iter(observation["lane_observations"][lane]["result_json"]["summary.json"]["arms"].values()))
        arm.update(updates=7, training_exposures=79, calibration_queries=7, heldout_queries=79,
                   c80_queries=79, all_parameters_frozen=False, formal_infer_max_abs=2e-6,
                   orthogonality={"action": 6e-5, "video": 1e-7})
        training_arm = next(iter(observation["lane_observations"][lane]["result_json"]["training.json"]["arms"].values()))
        training_arm["history"].pop()
        observation["lane_observations"][lane]["result_json"]["summary.json"]["new_teacher_queries"] = 79
        failures = control.verification_failures(observation, protocol, "coverage")
        for expected in ("updates", "training_exposures", "calibration_queries", "heldout_queries",
                         "c80_queries", "frozen parameters", "formal inference parity", "orthogonality",
                         "training history", "new_teacher_queries"):
            self.assertTrue(any(expected in failure for failure in failures), expected)

    def test_terminal_artifact_receipts_are_required(self):
        protocol, observation = passing_observation("path")
        observation["scheduler"].update(job_state="R", Exit_status="1")
        observation.update(artifact_exit_code=1, phase_complete=False)
        observation["lane_observations"]["path"].update(artifact_exit_code=1, phase_complete=False)
        failures = control.verification_failures(observation, protocol, "path")
        for expected in ("terminal F/X", "Exit_status", "exit_code.txt", "PHASE_COMPLETE"):
            self.assertTrue(any(expected in failure for failure in failures), expected)

    def test_reused_endpoint_replay_is_checked(self):
        protocol, observation = passing_observation("path")
        reused = next(iter(observation["result_json"]["summary.json"]["arms"].values()))
        reused["source_replay_metric_max_abs"] = 3e-6
        failures = control.verification_failures(observation, protocol, "path")
        self.assertTrue(any("source replay" in failure for failure in failures))


class SubmissionSafetyTests(unittest.TestCase):
    def test_each_job_uses_normal_route_and_verified_one_gpu_g1_resources(self):
        for kind in control.KINDS:
            command = control.qsub_command(kind, control.job_name(kind, 0), 0)
            with self.subTest(kind=kind):
                self.assertIn("-q normal", command)
                self.assertIn("select=1:ncpus=16:mem=110gb:ngpus=1", command)
                self.assertIn("walltime=04:00:00", command)
                self.assertNotIn("-q g1", command)
                self.assertEqual(control.job_resources(kind)[0], "g1")

    def test_existing_handle_is_returned_without_any_qsub_or_status_command(self):
        state = {"handle": {"job_id": "25900001.pbs101", "remote_root": control.ROOT,
                            "protocol": control.PROTOCOL_ID, "job_kind": "path"}}

        class FakeRemote:
            def sftp_client(self, _transport):
                return nullcontext(object())

            def command(self, *_args, **_kwargs):
                raise AssertionError("existing handle must not trigger another remote command")

        with patch.object(control, "load_protocol", return_value={}), \
             patch.object(control, "read_remote_json", return_value={}), \
             patch.object(control, "read_local_state", return_value=state):
            result = control.submit(FakeRemote(), object(), "path")
        self.assertTrue(result["existing"])

    def test_unresolved_qsub_intent_never_submits_again(self):
        class FakeRemote:
            def __init__(self):
                self.commands = []

            def command(self, _transport, command, timeout):
                self.commands.append(command)
                return 0, ""

        remote = FakeRemote()
        state = {"intent": {"kind": "submit", "status": "intent_before_qsub",
                            "job_name": control.job_name("path", 0), "attempt": 0}}
        with self.assertRaisesRegex(RuntimeError, "Do not resubmit"):
            control.recover_intent(remote, object(), state, "path")
        self.assertEqual(len(remote.commands), 1)
        self.assertIn("qselect -x", remote.commands[0])
        self.assertNotIn("qsub", remote.commands[0])

    def test_query_timeout_cannot_authorize_resume(self):
        protocol, observation = passing_observation("path")
        observation["scheduler"].update(query_rc=1, job_state="R", Exit_status=None)
        observation["artifact_exit_code"] = 75
        observation["phase_complete"] = False
        lane = observation["lane_observations"]["path"]
        lane["artifact_exit_code"] = 75
        lane["phase_complete"] = False
        summary = lane["result_json"]["summary.json"]
        summary.update(status="checkpointed", complete=False)
        lane["result_json"]["progress.json"].update(status="checkpointed", complete=False)
        failures = control.resume_failures(observation, protocol, "path")
        self.assertTrue(any("query timeout" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
