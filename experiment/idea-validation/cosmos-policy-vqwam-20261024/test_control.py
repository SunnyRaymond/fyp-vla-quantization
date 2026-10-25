"""Small, CPU-only control receipt tests; no SSH or PBS calls are required."""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

import control


class FakeSftp:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def file(self, path, mode):
        return BytesIO()

    def posix_rename(self, source, target):
        pass


class FakeRemote:
    def __init__(self, statuses):
        self.statuses = statuses
        self.commands = []

    def command(self, transport, command, timeout=0):
        self.commands.append(command)
        if command.startswith("qstat -xf "):
            job_id = command.split()[-1]
            return self.statuses.get(job_id, (153, "Unknown Job Id"))
        if command.startswith("qselect "):
            return 153, ""
        if command.startswith("test -f "):
            return 0, ""
        if command.startswith("qsub "):
            return 0, "333.pbs101\n"
        return 0, ""

    def sftp_client(self, transport):
        return FakeSftp()


class _MemoryWriteBuffer(BytesIO):
    def __init__(self, remote, path):
        super().__init__()
        self.remote, self.path = remote, path

    def close(self):
        if not self.closed:
            self.remote.files[self.path] = self.getvalue()
        super().close()


class MemorySftp:
    def __init__(self, remote):
        self.remote = remote

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def file(self, path, mode):
        if "r" in mode:
            if hasattr(self.remote, "events") and path.endswith("/submissions_ready.json"):
                self.remote.events.append(("read_pair", path))
            return BytesIO(self.remote.files[path])
        return _MemoryWriteBuffer(self.remote, path)

    def posix_rename(self, source, target):
        self.remote.files[target] = self.remote.files.pop(source)


class MemoryRemote:
    LIMIT = 256 * 1024

    def __init__(self, statuses, files=None):
        self.statuses = statuses
        self.files = dict(files or {})
        self.commands = []
        self.events = []
        self.next_jobs = {"cpvq24osht": "404.pbs101", "cpvq24olon": "405.pbs101"}

    def command(self, transport, command, timeout=0):
        self.commands.append(command)
        parts = shlex.split(command)
        if parts[:2] == ["qstat", "-xf"]:
            return self.statuses.get(parts[-1], (153, "Unknown Job Id"))
        if parts and parts[0] == "qselect":
            return 153, ""
        if parts[:2] == ["test", "-e"]:
            return (0, "exists") if parts[2] in self.files else (1, "")
        if parts and parts[0] == "qsub":
            name = parts[parts.index("-N") + 1]
            return 0, self.next_jobs[name] + "\n"
        return 0, ""

    def sftp_client(self, transport):
        return MemorySftp(self)


class StartupRepairRemote(MemoryRemote):
    def command(self, transport, command, timeout=0):
        parts = shlex.split(command)
        if parts and parts[0] == "qsub":
            self.commands.append(command)
            name = parts[parts.index("-N") + 1]
            job_id = self.next_jobs[name]
            self.statuses[job_id] = (0, f"Job_Name = {name}\njob_state = H\nExit_status = None\n")
            return 0, job_id + "\n"
        return super().command(transport, command, timeout)


class PreclaimResumeRemote(MemoryRemote):
    qrls_return_code = 0

    def command(self, transport, command, timeout=0):
        parts = shlex.split(command)
        if parts and parts[0] == "tail":
            self.commands.append(command)
            path = parts[-1]
            return (0, self.files[path].decode(errors="replace")) if path in self.files else (1, "")
        if parts and parts[0] == "qsub":
            self.commands.append(command)
            name = parts[parts.index("-N") + 1]
            job_id = self.next_jobs[name]
            self.events.append(("qsub", command))
            self.statuses[job_id] = (0, f"Job_Name = {name}\njob_state = H\nExit_status = None\n")
            return 0, job_id + "\n"
        if parts and parts[0] == "qrls":
            self.commands.append(command)
            job_id = parts[1]
            self.events.append(("qrls", job_id))
            self.release_pair_at_qrls = json.loads(self.files[
                self.protocol_root + "/control/ours_race/submissions_ready.json"])
            self.release_receipt_at_qrls = json.loads(self.files[
                self.protocol_root + "/control/ours_race/resume_release_" +
                job_id.replace(".", "_") + ".json"])
            if self.qrls_return_code == 0:
                name = self.statuses[job_id][1].split("Job_Name = ", 1)[1].splitlines()[0]
                self.statuses[job_id] = (0, f"Job_Name = {name}\njob_state = Q\nExit_status = None\n")
            return self.qrls_return_code, "simulated qrls result"
        return super().command(transport, command, timeout)


class ControlContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = control.load_protocol()

    def test_all_frozen_stages_have_pbs_wrappers(self):
        stage_map = control.stages(self.protocol)
        self.assertEqual(set(stage_map), set(control.DEFAULT_STAGES))
        for _, script in stage_map.values():
            self.assertTrue((control.HERE / script).is_file(), script)

    def test_resource_caps_and_selected_evaluation_route(self):
        for stage in ("perf_bf16", "perf_spinquant", "perf_quarot", "perf_svdquant", "perf_viditq", "perf_ours",
                      "quarot", "svdquant", "viditq", "ours_short"):
            resource = control.resources(self.protocol, stage)
            self.assertLess(resource["walltime"], "02:00:00")
            self.assertEqual(resource["gpus"], 1)
        self.assertEqual(control.resources(self.protocol, "ours_long")["walltime"], "12:00:00")
        evaluation = control.resources(self.protocol, "evaluation")
        self.assertIn((evaluation["gpus"], evaluation["ncpus"], evaluation["memory_gb"],
                       evaluation['required_execution_queue']),
                      ((4, 64, 440, 'g3'), (8, 128, 880, 'g4')))
        self.assertEqual(evaluation["walltime"], "24:00:00")
        self.assertEqual(evaluation["queue"], "normal")
        extraction = control.resources(self.protocol, 'result_extract')
        self.assertEqual((extraction['gpus'], extraction['ncpus'], extraction['memory_gb'],
                          extraction['walltime']), (0, 4, 16, '00:20:00'))
        followup = control.resources(self.protocol, "baseline_followup_bundle")
        self.assertEqual((followup["gpus"], followup["ncpus"], followup["memory_gb"],
                          followup["walltime"], followup["queue"],
                          followup["required_execution_queue"]),
                         (2, 32, 220, "04:00:00", "normal", "g2"))
        self.assertEqual(control.stages(self.protocol)["baseline_followup_bundle"],
                         ("cpvq24c2", "baseline_followup_bundle.pbs"))

    def test_evaluation_submission_accepts_only_matched_four_or_eight_gpu_profiles(self):
        protocol = json.loads(json.dumps(self.protocol))
        protocol["resources"]["evaluation"].update({
            "gpus": 8, "ncpus": 128, "memory_gb": 880, "nodes": 2,
            "gpus_per_node": 4, "ncpus_per_node": 64, "memory_gb_per_node": 440,
            "walltime": "24:00:00",
            "queue": "normal", "required_execution_queue": "g4",
        })
        protocol["load_balancing"]["persistent_workers"] = 8
        protocol["stages"] = {"evaluation": {
            "job_name": "cpvq24eval", "script": "evaluation_g4.pbs"}}
        value = control.resources(protocol, "evaluation")
        self.assertEqual((value["gpus"], value["ncpus"], value["memory_gb"], value["nodes"]),
                         (8, 128, 880, 2))
        self.assertEqual(control.stages(protocol)["evaluation"],
                         ("cpvq24eval", "evaluation_g4.pbs"))

        protocol["resources"]["evaluation"]["required_execution_queue"] = "g3"
        with self.assertRaisesRegex(ValueError, "exact 4-GPU/g3 or 8-GPU/two-node g4"):
            control.resources(protocol, "evaluation")

    def test_evaluation_g4_qsub_uses_two_scattered_four_gpu_chunks(self):
        protocol = json.loads(json.dumps(self.protocol))
        protocol["resources"]["evaluation"].update({
            "gpus": 8, "ncpus": 128, "memory_gb": 880, "nodes": 2,
            "gpus_per_node": 4, "ncpus_per_node": 64, "memory_gb_per_node": 440,
            "walltime": "24:00:00", "queue": "normal", "required_execution_queue": "g4",
        })
        protocol["load_balancing"]["persistent_workers"] = 8
        protocol["stages"] = {"evaluation": {
            "job_name": "cpvq24eval", "script": "evaluation_g4.pbs"}}
        with tempfile.TemporaryDirectory() as directory, patch.object(control, "HERE", Path(directory)):
            remote = FakeRemote({})
            result = control.submit(remote, object(), protocol, "evaluation")
            command = next(command for command in remote.commands if command.startswith("qsub "))
            self.assertIn("select=2:ncpus=64:mem=440gb:ngpus=4:mpiprocs=1", command)
            self.assertIn("-l place=scatter", command)
            self.assertTrue(command.endswith("evaluation_g4.pbs"))
            self.assertEqual(result["resources"]["nodes"], 2)

    def test_evaluation_wrapper_reads_protocol_profile_and_has_g4_pbs_entrypoint(self):
        script = (control.HERE / "job.sh").read_text(encoding="utf-8")
        self.assertIn('"$ROOT/protocol.json"', script)
        self.assertIn('"EVALUATION_PROFILE queue=$expected_queue nodes=$expected_nodes gpus=$expected_gpus workers=$expected_workers local_gpus=$gpus_per_node"', script)
        self.assertNotIn("Four-GPU evaluation was not routed to g3", script)
        pbs = (control.HERE / "evaluation_g4.pbs").read_text(encoding="utf-8")
        self.assertIn("#PBS -N cpvq24eval", pbs)
        self.assertIn("select=2:ncpus=64:mem=440gb:ngpus=4:mpiprocs=1", pbs)
        self.assertIn("#PBS -l place=scatter", pbs)
        self.assertIn("#PBS -q normal", pbs)
        self.assertIn("#PBS -l walltime=24:00:00", pbs)
        self.assertEqual(control.summary_relative_path(self.protocol, "baseline_followup_bundle"),
                         "results/baseline_followup_bundle/summary.json")

    def test_baseline_followup_job_gate_is_scoped_to_registered_candidates(self):
        script = (control.HERE / "job.sh").read_text(encoding="utf-8")
        self.assertIn("perf_svdquant|vidcal|perf_viditq", script)
        self.assertIn("--race-directory control/baseline_followup_race", script)
        viditq_perf = script.split("    perf_viditq)\n", 1)[1].split(";;", 1)[0]
        self.assertIn('if [[ "${COSMOS_PARENT_BUNDLE:-}" == cpvq24c2 ]]; then', viditq_perf)
        self.assertNotIn("--candidate single", viditq_perf)

    def test_existing_live_handle_is_returned_without_another_submission(self):
        stage = "ours_short"
        name = control.stages(self.protocol)[stage][0]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(control, "HERE", Path(directory)):
                (Path(directory) / f"{stage}_handle.json").write_text(json.dumps({
                    "job_id": "12345.pbs101", "job_name": name,
                }), encoding="utf-8")

                class Remote:
                    calls = []

                    @classmethod
                    def command(cls, transport, command, timeout=0):
                        cls.calls.append(command)
                        return 0, f"Job_Name = {name}\njob_state = R\nExit_status = None\n"

                result = control.submit(Remote, object(), self.protocol, stage)
                self.assertEqual(result["job_state"], "R")
                self.assertEqual(Remote.calls, ["qstat -xf 12345.pbs101"])

    def test_unresolved_intent_blocks_a_duplicate_qsub(self):
        stage = "teacher"
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(control, "HERE", Path(directory)):
                (Path(directory) / f"{stage}_intent.json").write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "Durable submission intent"):
                    control.submit(object(), object(), self.protocol, stage)

    def test_checkpoint_resume_paths_follow_stage_outputs(self):
        self.assertEqual(control.summary_relative_path(self.protocol, "prepare"),
                         "results/preparation/summary.json")
        self.assertEqual(control.summary_relative_path(self.protocol, "quarot"),
                         "results/quarot/summary.json")
        self.assertEqual(control.summary_relative_path(self.protocol, "perf_ours"),
                         "results/performance/ours_vqwam/summary.json")
        self.assertEqual(control.summary_relative_path(self.protocol, "ours_short"),
                         "results/ours/summary.json")
        self.assertEqual(control.summary_relative_path(self.protocol, "result_extract"),
                         "results/reporting/summary.json")

    def test_checkpoint_resume_requires_same_protocol_and_job(self):
        summary = {"protocol": self.protocol["protocol"], "status": "checkpointed",
                   "pbs_job_id": "12345.pbs101", "arm": "quarot_w4a4"}
        control.validate_checkpointed_summary(
            self.protocol, "quarot", summary, {"job_id": "12345.pbs101"})
        summary["pbs_job_id"] = "99999.pbs101"
        with self.assertRaisesRegex(RuntimeError, "different PBS job"):
            control.validate_checkpointed_summary(
                self.protocol, "quarot", summary, {"job_id": "12345.pbs101"})
        summary["pbs_job_id"] = None
        with self.assertRaisesRegex(RuntimeError, "different PBS job"):
            control.validate_checkpointed_summary(
                self.protocol, "quarot", summary, {"job_id": "12345.pbs101"})

    def test_prepare_module_is_loaded_after_gcc_and_invoked_by_absolute_path(self):
        script = (control.HERE / "job.sh").read_text(encoding="utf-8")
        prepare = script.split('if [[ "$STAGE" == prepare ]]; then', 1)[1].split("\n  fi", 1)[0]
        self.assertLess(prepare.index("module load gcc/11.4.0-nscc"),
                        prepare.index("module load python/3.11.7-gcc11"))
        self.assertIn('MODULE_PYTHON=$(type -P python3.11)', prepare)
        self.assertIn('"$MODULE_PYTHON" "$ARTIFACTS/control_sources/prepare.py"', prepare)
        self.assertNotIn('python3 "$ARTIFACTS/control_sources/prepare.py"', prepare)

    def test_failed_stage_repair_requires_nonzero_exit_and_records_durable_intent(self):
        stage = "prepare"
        name = control.stages(self.protocol)[stage][0]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(control, "HERE", Path(directory)):
                previous = {"stage": stage, "job_name": name, "job_id": "101.pbs101", "sequence": 0}
                (Path(directory) / f"{stage}_handle.json").write_text(json.dumps(previous), encoding="utf-8")
                remote = FakeRemote({"101.pbs101": (0, f"Job_Name = {name}\njob_state = F\nExit_status = 1\n")})
                result = control.submit(remote, object(), self.protocol, stage,
                                        repair_note="prepare failed before the module fix")
                self.assertEqual(result["repair_kind"], "nonzero_exit")
                self.assertEqual(result["repair_note"], "prepare failed before the module fix")
                self.assertEqual(result["job_id"], "333.pbs101")
                self.assertTrue((Path(directory) / "prepare_r0_handle.json").is_file())
                intent = json.loads((Path(directory) / "prepare_r1_intent.json").read_text(encoding="utf-8"))
                self.assertEqual(intent["state"], "submitted")
                self.assertEqual(intent["previous_job_id"], "101.pbs101")
                alias_intent = json.loads((Path(directory) / "prepare_intent.json").read_text(encoding="utf-8"))
                self.assertEqual(alias_intent["job_id"], "333.pbs101")

    def test_never_started_repair_requires_exact_failed_parent_and_new_dependency(self):
        stage, parent_stage = "teacher", "prepare"
        name = control.stages(self.protocol)[stage][0]
        parent_name = control.stages(self.protocol)[parent_stage][0]
        child_id, parent_id = "202.pbs101", "101.pbs101"
        child_status = f"Job_Name = {name}\njob_state = F\ndepend = afterok:{parent_id}@pbs101\n"
        parent_status = f"Job_Name = {parent_name}\njob_state = F\nExit_status = 1\n"
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(control, "HERE", Path(directory)):
                (Path(directory) / f"{parent_stage}_handle.json").write_text(json.dumps({
                    "stage": parent_stage, "job_name": parent_name, "job_id": parent_id,
                }), encoding="utf-8")
                (Path(directory) / f"{stage}_handle.json").write_text(json.dumps({
                    "stage": stage, "job_name": name, "job_id": child_id,
                    "dependency": parent_id, "sequence": 0,
                }), encoding="utf-8")
                remote = FakeRemote({child_id: (0, child_status), parent_id: (0, parent_status)})
                result = control.submit(remote, object(), self.protocol, stage,
                                        dependency="303.pbs101", repair_never_started=True)
                self.assertEqual(result["repair_kind"], "never_started_dependency_terminal_exit_unavailable")
                self.assertEqual(result["repair_source_dependency"], parent_id)
                self.assertEqual(result["dependency"], "303.pbs101")
                proof = result["repair_proof"]
                self.assertEqual(proof["root_failed_ancestor"]["job_id"], parent_id)
                self.assertIn("explicit nonzero", proof["summary"])
                self.assertIn("never_started_dependency_terminal", result["repair_kind"])

    def test_never_started_repair_traces_wrapped_afterok_cascade_and_records_proof(self):
        child_stage, bundle_stage, cache_stage = "teacher", "bootstrap_bundle", "language_cache"
        stage_map = control.stages(self.protocol)
        child_id, bundle_id, cache_id = "202.pbs101", "101.pbs101", "100.pbs101"
        child_name = stage_map[child_stage][0]
        bundle_name = stage_map[bundle_stage][0]
        cache_name = stage_map[cache_stage][0]
        child_status = (f"Job_Name = {child_name}\njob_state = F\n"
                        f"depend = afterok:{bundle_id}@pbs101,\n"
                        "        beforeok:303.pbs101@pbs101,\n")
        bundle_status = (f"Job_Name = {bundle_name}\njob_state = X\n"
                         f"depend = afterok:{cache_id}@pbs101,\n"
                         f"        beforeok:{child_id}@pbs101,\n")
        cache_status = f"Job_Name = {cache_name}\njob_state = F\nExit_status = 1\n"
        with tempfile.TemporaryDirectory() as directory, patch.object(control, "HERE", Path(directory)):
            (Path(directory) / f"{child_stage}_handle.json").write_text(json.dumps({
                "stage": child_stage, "job_name": child_name, "job_id": child_id,
                "dependency": bundle_id, "sequence": 0,
            }), encoding="utf-8")
            (Path(directory) / f"{bundle_stage}_handle.json").write_text(json.dumps({
                "stage": bundle_stage, "job_name": bundle_name, "job_id": bundle_id,
                "dependency": cache_id, "sequence": 0,
            }), encoding="utf-8")
            (Path(directory) / f"{cache_stage}_handle.json").write_text(json.dumps({
                "stage": cache_stage, "job_name": cache_name, "job_id": cache_id,
            }), encoding="utf-8")
            remote = FakeRemote({child_id: (0, child_status), bundle_id: (0, bundle_status),
                                 cache_id: (0, cache_status)})
            parsed = control.parse_status(remote, object(), child_id)
            self.assertEqual(parsed["depend"],
                             f"afterok:{bundle_id}@pbs101,beforeok:303.pbs101@pbs101,")
            result = control.submit(remote, object(), self.protocol, child_stage,
                                    dependency="303.pbs101", repair_never_started=True)
            self.assertEqual(result["repair_kind"],
                             "never_started_dependency_cascade_terminal_exit_unavailable")
            proof = result["repair_proof"]
            self.assertEqual([row["job_id"] for row in proof["ancestry"]], [bundle_id, cache_id])
            self.assertEqual(proof["root_failed_ancestor"]["exit_status"], 1)
            self.assertIn("1 never-started intermediate", proof["summary"])

    def test_never_started_cascade_fails_closed_on_live_intermediate_or_missing_root_exit(self):
        stage_map = control.stages(self.protocol)
        child_id, bundle_id, cache_id = "202.pbs101", "101.pbs101", "100.pbs101"
        child_name, bundle_name, cache_name = (stage_map["teacher"][0],
                                               stage_map["bootstrap_bundle"][0],
                                               stage_map["language_cache"][0])
        child_status = f"Job_Name = {child_name}\njob_state = F\ndepend = afterok:{bundle_id}\n"
        handles = {
            "teacher": {"stage": "teacher", "job_name": child_name,
                        "job_id": child_id, "dependency": bundle_id},
            "bootstrap_bundle": {"stage": "bootstrap_bundle", "job_name": bundle_name,
                                 "job_id": bundle_id, "dependency": cache_id},
            "language_cache": {"stage": "language_cache", "job_name": cache_name,
                               "job_id": cache_id},
        }
        scenarios = (
            ("live intermediate", f"Job_Name = {bundle_name}\njob_state = R\ndepend = afterok:{cache_id}\n",
             f"Job_Name = {cache_name}\njob_state = F\nExit_status = 1\n", "intermediary"),
            ("intermediate exited", f"Job_Name = {bundle_name}\njob_state = F\nExit_status = 0\n",
             f"Job_Name = {cache_name}\njob_state = F\nExit_status = 1\n", "intermediary"),
            ("root exit missing", f"Job_Name = {bundle_name}\njob_state = X\ndepend = afterok:{cache_id}\n",
             f"Job_Name = {cache_name}\njob_state = F\n", "recorded parent"),
            ("intermediate walltime", f"Job_Name = {bundle_name}\njob_state = X\ndepend = afterok:{cache_id}\nresources_used.walltime = 00:00:00\n",
             f"Job_Name = {cache_name}\njob_state = F\nExit_status = 1\n", "intermediary"),
        )
        for label, bundle_status, cache_status, error in scenarios:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory, \
                    patch.object(control, "HERE", Path(directory)):
                for stage, handle in handles.items():
                    (Path(directory) / f"{stage}_handle.json").write_text(
                        json.dumps(handle), encoding="utf-8")
                remote = FakeRemote({
                    child_id: (0, child_status), bundle_id: (0, bundle_status),
                    cache_id: (0, cache_status),
                })
                with self.assertRaisesRegex(RuntimeError, error):
                    control.submit(remote, object(), self.protocol, "teacher",
                                   dependency="303.pbs101", repair_never_started=True)
                self.assertFalse(any(command.startswith("qsub ") for command in remote.commands))

    def test_never_started_repair_rejects_multiple_afterok_references(self):
        stage, parent_stage = "teacher", "prepare"
        name, parent_name = control.stages(self.protocol)[stage][0], control.stages(self.protocol)[parent_stage][0]
        child_id, parent_id = "202.pbs101", "101.pbs101"
        with tempfile.TemporaryDirectory() as directory, patch.object(control, "HERE", Path(directory)):
            (Path(directory) / f"{parent_stage}_handle.json").write_text(json.dumps({
                "stage": parent_stage, "job_name": parent_name, "job_id": parent_id,
            }), encoding="utf-8")
            (Path(directory) / f"{stage}_handle.json").write_text(json.dumps({
                "stage": stage, "job_name": name, "job_id": child_id,
                "dependency": parent_id,
            }), encoding="utf-8")
            remote = FakeRemote({
                child_id: (0, f"Job_Name = {name}\njob_state = F\n"
                              f"depend = afterok:{parent_id},afterok:999.pbs101,\n"),
                parent_id: (0, f"Job_Name = {parent_name}\njob_state = F\nExit_status = 1\n"),
            })
            with self.assertRaisesRegex(RuntimeError, "one exact afterok reference"):
                control.submit(remote, object(), self.protocol, stage,
                               dependency="303.pbs101", repair_never_started=True)
            self.assertFalse(any(command.startswith("qsub ") for command in remote.commands))

    def test_never_started_cascade_rejects_unknown_or_cyclic_ancestor(self):
        stage_map = control.stages(self.protocol)
        child_id, bundle_id, cache_id = "202.pbs101", "101.pbs101", "100.pbs101"
        child_name, bundle_name, cache_name = (stage_map["teacher"][0],
                                               stage_map["bootstrap_bundle"][0],
                                               stage_map["language_cache"][0])
        child_status = f"Job_Name = {child_name}\njob_state = F\ndepend = afterok:{bundle_id}\n"
        cases = (
            ("unknown", False,
             f"Job_Name = {bundle_name}\njob_state = X\ndepend = afterok:{cache_id}\n",
             "Recorded dependency has no identity-matched"),
            ("cycle", True,
             f"Job_Name = {bundle_name}\njob_state = X\ndepend = afterok:{cache_id}\n",
             "cycle"),
        )
        for label, include_cache, bundle_status, error in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory, \
                    patch.object(control, "HERE", Path(directory)):
                handles = {
                    "teacher": {"stage": "teacher", "job_name": child_name,
                                "job_id": child_id, "dependency": bundle_id},
                    "bootstrap_bundle": {"stage": "bootstrap_bundle", "job_name": bundle_name,
                                         "job_id": bundle_id, "dependency": cache_id},
                }
                statuses = {child_id: (0, child_status), bundle_id: (0, bundle_status)}
                if include_cache:
                    handles["language_cache"] = {"stage": "language_cache", "job_name": cache_name,
                                                 "job_id": cache_id, "dependency": bundle_id}
                    statuses[cache_id] = (0, f"Job_Name = {cache_name}\njob_state = F\n"
                                             f"depend = afterok:{bundle_id}\n")
                for handle_stage, handle in handles.items():
                    (Path(directory) / f"{handle_stage}_handle.json").write_text(
                        json.dumps(handle), encoding="utf-8")
                remote = FakeRemote(statuses)
                with self.assertRaisesRegex(RuntimeError, error):
                    control.submit(remote, object(), self.protocol, "teacher",
                                   dependency="303.pbs101", repair_never_started=True)
                self.assertFalse(any(command.startswith("qsub ") for command in remote.commands))

    def test_never_started_repair_fails_closed_without_qstat_dependency_proof(self):
        stage, parent_stage = "teacher", "prepare"
        name = control.stages(self.protocol)[stage][0]
        parent_name = control.stages(self.protocol)[parent_stage][0]
        child_id, parent_id = "202.pbs101", "101.pbs101"
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(control, "HERE", Path(directory)):
                (Path(directory) / f"{parent_stage}_handle.json").write_text(json.dumps({
                    "stage": parent_stage, "job_name": parent_name, "job_id": parent_id,
                }), encoding="utf-8")
                (Path(directory) / f"{stage}_handle.json").write_text(json.dumps({
                    "stage": stage, "job_name": name, "job_id": child_id,
                    "dependency": parent_id, "sequence": 0,
                }), encoding="utf-8")
                remote = FakeRemote({
                    child_id: (0, f"Job_Name = {name}\njob_state = F\n"),
                    parent_id: (0, f"Job_Name = {parent_name}\njob_state = F\nExit_status = 1\n"),
                })
                with self.assertRaisesRegex(RuntimeError, "qstat depend"):
                    control.submit(remote, object(), self.protocol, stage,
                                   dependency="303.pbs101", repair_never_started=True)

    def _ours_pair_repair_fixture(self, directory, occupied=None):
        protocol = self.protocol
        old_parent, new_parent = "101.pbs101", "303.pbs101"
        short_id, long_id = "201.pbs101", "202.pbs101"
        stage_map = control.stages(protocol)
        old_parent_name = stage_map["bootstrap_bundle"][0]
        files = {}
        if occupied:
            files[protocol["remote_root"] + "/control/ours_race/" + occupied] = b"{}"
        pair = {
            "protocol": protocol["protocol"],
            "jobs": {
                "short": {"job_id": short_id, "job_name": stage_map["ours_short"][0]},
                "long": {"job_id": long_id, "job_name": stage_map["ours_long"][0]},
            },
            "short": {"job_id": short_id, "job_name": stage_map["ours_short"][0]},
            "long": {"job_id": long_id, "job_name": stage_map["ours_long"][0]},
            "created_utc": "2026-10-24T00:00:00+00:00",
        }
        pair_path = Path(directory) / "ours_race" / "submissions_ready.json"
        pair_path.parent.mkdir(parents=True)
        pair_path.write_text(json.dumps(pair), encoding="utf-8")
        root = protocol["remote_root"]
        files[root + "/control/ours_race/submissions_ready.json"] = json.dumps(pair).encode()

        statuses = {
            short_id: (0, f"Job_Name = {stage_map['ours_short'][0]}\njob_state = F\n"
                          f"depend = afterok:{old_parent}@pbs101\n"),
            long_id: (0, f"Job_Name = {stage_map['ours_long'][0]}\njob_state = X\n"
                         f"depend = afterok:{old_parent}@pbs101\n"),
            old_parent: (0, f"Job_Name = {old_parent_name}\njob_state = F\nExit_status = 1\n"),
            new_parent: (0, f"Job_Name = {old_parent_name}\njob_state = R\n"),
        }
        handles = {
            "ours_short": {"stage": "ours_short", "job_name": stage_map["ours_short"][0],
                           "job_id": short_id, "dependency": old_parent, "sequence": 0},
            "ours_long": {"stage": "ours_long", "job_name": stage_map["ours_long"][0],
                          "job_id": long_id, "dependency": old_parent, "sequence": 0},
        }
        for stage, handle in handles.items():
            (Path(directory) / f"{stage}_handle.json").write_text(json.dumps(handle), encoding="utf-8")
        (Path(directory) / "bootstrap_bundle_r0_handle.json").write_text(json.dumps({
            "stage": "bootstrap_bundle", "job_name": old_parent_name, "job_id": old_parent,
        }), encoding="utf-8")
        (Path(directory) / "bootstrap_bundle_handle.json").write_text(json.dumps({
            "stage": "bootstrap_bundle", "job_name": old_parent_name, "job_id": new_parent,
        }), encoding="utf-8")
        return MemoryRemote(statuses, files), pair, old_parent, new_parent

    def test_ours_pair_dependency_repair_archives_and_publishes_only_after_both_new_ids(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(control, "HERE", Path(directory)):
            remote, old_pair, old_parent, new_parent = self._ours_pair_repair_fixture(directory)
            result = control.repair_ours_pair_dependency(remote, object(), self.protocol, new_parent)

            self.assertEqual(result["previous_dependency"], old_parent)
            self.assertEqual(result["repaired_dependency"], new_parent)
            self.assertEqual(result["jobs"]["short"]["job_id"], "404.pbs101")
            self.assertEqual(result["jobs"]["long"]["job_id"], "405.pbs101")
            archive = json.loads(Path(result["local_archive"]).read_text(encoding="utf-8"))
            self.assertEqual(archive, old_pair)
            ready_path = Path(directory) / "ours_race" / "submissions_ready.json"
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            self.assertEqual(ready["dependency"], new_parent)
            self.assertEqual(ready["short"]["job_id"], "404.pbs101")
            self.assertEqual(ready["long"]["job_id"], "405.pbs101")
            remote_ready = json.loads(remote.files[
                self.protocol["remote_root"] + "/control/ours_race/submissions_ready.json"])
            self.assertEqual(remote_ready, ready)
            self.assertEqual(sum(command.startswith("qsub ") for command in remote.commands), 2)
            self.assertTrue(all("depend=afterok:" + new_parent in command
                                for command in remote.commands if command.startswith("qsub ")))
            self.assertFalse(any("qdel" in command or "qalter" in command for command in remote.commands))
            for stage in ("ours_short", "ours_long"):
                handle = json.loads((Path(directory) / f"{stage}_handle.json").read_text(encoding="utf-8"))
                self.assertEqual(handle["dependency"], new_parent)
                self.assertEqual(handle["repair_source_dependency"], old_parent)

    def test_ours_pair_dependency_repair_refuses_any_race_claim_before_submission(self):
        for occupied in ("winner.json", "claim", "resume_claim"):
            with self.subTest(occupied=occupied), tempfile.TemporaryDirectory() as directory, \
                    patch.object(control, "HERE", Path(directory)):
                remote, _, _, new_parent = self._ours_pair_repair_fixture(directory, occupied=occupied)
                with self.assertRaisesRegex(RuntimeError, "already has"):
                    control.repair_ours_pair_dependency(remote, object(), self.protocol, new_parent)
                self.assertFalse(any(command.startswith("qsub ") for command in remote.commands))

    def test_ours_pair_dependency_repair_rejects_started_or_exited_candidate(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(control, "HERE", Path(directory)):
            remote, _, _, new_parent = self._ours_pair_repair_fixture(directory)
            name = control.stages(self.protocol)["ours_short"][0]
            remote.statuses["201.pbs101"] = (0, f"Job_Name = {name}\njob_state = F\n"
                                                    "Exit_status = 1\n")
            with self.assertRaisesRegex(RuntimeError, "missing Exit_status"):
                control.repair_ours_pair_dependency(remote, object(), self.protocol, new_parent)
            self.assertFalse(any(command.startswith("qsub ") for command in remote.commands))

    def test_never_started_ours_repair_cannot_be_called_through_generic_submit(self):
        with self.assertRaisesRegex(RuntimeError, "existing race-resume path"):
            control.submit(object(), object(), self.protocol, "ours_short", dependency="303.pbs101",
                           repair_never_started=True)

    def _ours_startup_repair_fixture(self, directory, *, summary=None, checkpoint=False,
                                     sibling_state="X", qselect_output=""):
        protocol = self.protocol
        stage_map = control.stages(protocol)
        old_id, sibling_id = "601.pbs101", "602.pbs101"
        winner_stage, sibling_stage = "ours_short", "ours_long"
        winner_name, sibling_name = stage_map[winner_stage][0], stage_map[sibling_stage][0]
        winner = {"stage": winner_stage, "job_id": old_id, "job_name": winner_name}
        pair = {
            "protocol": protocol["protocol"],
            "jobs": {"short": {"job_id": old_id, "job_name": winner_name},
                     "long": {"job_id": sibling_id, "job_name": sibling_name}},
            "short": {"job_id": old_id, "job_name": winner_name},
            "long": {"job_id": sibling_id, "job_name": sibling_name},
            "dependency": "500.pbs101",
        }
        race_root = protocol["remote_root"] + "/control/ours_race"
        files = {
            race_root + "/winner.json": json.dumps(winner).encode(),
            race_root + "/submissions_ready.json": json.dumps(pair).encode(),
            race_root + "/claim": b"",
        }
        if summary is not None:
            files[protocol["remote_root"] + "/results/ours/summary.json"] = json.dumps(summary).encode()
        if checkpoint:
            files[protocol["remote_root"] + "/results/ours/checkpoint.pt"] = b"checkpoint"
        statuses = {
            old_id: (0, f"Job_Name = {winner_name}\njob_state = F\nExit_status = 1\n"
                        "stime = Sat Oct 24 11:30:00 2026\nexec_host = node01/0\n"
                        "resources_used.walltime = 00:00:46\n"),
            sibling_id: (0, f"Job_Name = {sibling_name}\njob_state = {sibling_state}\n"),
        }
        remote = StartupRepairRemote(statuses, files)
        if qselect_output:
            original_command = remote.command

            def command_with_occupied_name(transport, command, timeout=0):
                if command.startswith("qselect "):
                    remote.commands.append(command)
                    return 0, qselect_output
                return original_command(transport, command, timeout)

            remote.command = command_with_occupied_name
        race_dir = Path(directory) / "ours_race"
        race_dir.mkdir(parents=True)
        (race_dir / "submissions_ready.json").write_text(json.dumps(pair), encoding="utf-8")
        for stage, job_id, job_name, dep in (
                (winner_stage, old_id, winner_name, "500.pbs101"),
                (sibling_stage, sibling_id, sibling_name, "500.pbs101")):
            handle_text = json.dumps({
                "stage": stage, "job_id": job_id, "job_name": job_name,
                "sequence": 3, "dependency": dep,
            })
            (Path(directory) / f"{stage}_handle.json").write_text(handle_text, encoding="utf-8")
            (Path(directory) / f"{stage}_r3_handle.json").write_text(handle_text, encoding="utf-8")
            remote.files[protocol['remote_root'] + '/control/' + stage + '_r3_handle.json'] = handle_text.encode()
        return remote, old_id, sibling_id

    def test_ours_startup_repair_submits_only_failed_winner_held_and_publishes_pair(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(control, "HERE", Path(directory)):
            remote, old_id, sibling_id = self._ours_startup_repair_fixture(directory)
            result = control.repair_ours_startup(remote, object(), self.protocol,
                                                 "pinned tokenizer argv startup failure before update 1")

            self.assertEqual(result["stage"], "ours_short")
            self.assertEqual(result["old_winner_job_id"], old_id)
            self.assertEqual(result["sibling_job_id"], sibling_id)
            self.assertEqual(result["new_job_id_held"], "404.pbs101")
            self.assertTrue(result["requires_qrls"])
            qsub = next(command for command in remote.commands if command.startswith("qsub "))
            self.assertIn(" -h ", qsub)
            self.assertNotIn("depend=afterok:", qsub)
            self.assertEqual(sum(command.startswith("qsub ") for command in remote.commands), 1)
            marker = json.loads(remote.files[self.protocol["remote_root"] + "/control/ours_race/startup_repair.json"])
            self.assertEqual(marker["status"], "ready_for_release")
            self.assertEqual(marker["old_winner"]["job_id"], old_id)
            self.assertEqual(marker["new_winner"]["job_id"], "404.pbs101")
            self.assertTrue(marker["zero_updates"] and marker["no_checkpoint"])
            pair = json.loads(remote.files[self.protocol["remote_root"] + "/control/ours_race/submissions_ready.json"])
            self.assertEqual(pair["short"]["job_id"], "404.pbs101")
            self.assertEqual(pair["long"]["job_id"], sibling_id)
            intent = json.loads(remote.files[
                self.protocol["remote_root"] + "/control/ours_short_r4_intent.json"])
            self.assertEqual(intent["repair_kind"], "ours_startup_failure_retry")
            self.assertTrue(intent["initial_hold"])
            self.assertTrue(intent["repair_proof"]["no_checkpoint"])

    def test_ours_startup_repair_fails_closed_on_checkpoint_updates_or_started_sibling(self):
        cases = (
            ("checkpoint", {"checkpoint": True}, "checkpoint.pt exists"),
            ("completed updates", {"summary": {"protocol": self.protocol["protocol"],
                                                  "pbs_job_id": "601.pbs101",
                                                  "completed_updates": 1}}, "records optimizer"),
            ("rotation updates", {"summary": {"protocol": self.protocol["protocol"],
                                                "pbs_job_id": "601.pbs101",
                                                "rotation_completed": 1}}, "records optimizer"),
            ("running sibling", {"sibling_state": "R"}, "terminal and verified never started"),
            ("same-name job", {"qselect_output": "999.pbs101\n"}, "no live PBS job named"),
        )
        for label, overrides, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory, \
                    patch.object(control, "HERE", Path(directory)):
                remote, _, _ = self._ours_startup_repair_fixture(directory, **overrides)
                with self.assertRaisesRegex(RuntimeError, message):
                    control.repair_ours_startup(remote, object(), self.protocol, "startup failure")
                self.assertFalse(any(command.startswith("qsub ") for command in remote.commands))

    def test_ours_startup_repair_keeps_generic_ours_repair_note_disabled(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(control, "HERE", Path(directory)):
            self._ours_startup_repair_fixture(directory)
            with self.assertRaisesRegex(RuntimeError, "existing race-resume path"):
                control.submit(object(), object(), self.protocol, "ours_short",
                               repair_note="generic bypass")

    def _ours_preclaim_resume_fixture(self, directory, *, candidate_state="F", candidate_exit=64,
                                      previous_job_id="601.pbs101", winner_id="601.pbs101",
                                      candidate_job_id="603.pbs101",
                                      source_summary_id="601.pbs101", marker_owner="601.pbs101",
                                      startup_owner="601.pbs101", add_control_sources=False,
                                      log_text=None):
        protocol = self.protocol
        root = protocol["remote_root"]
        stage, sibling_stage = "ours_short", "ours_long"
        stage_name = control.stages(protocol)[stage][0]
        sibling_name = control.stages(protocol)[sibling_stage][0]
        source_id, sibling_id = "601.pbs101", "602.pbs101"
        candidate_id = candidate_job_id
        winner = {"stage": stage, "job_id": winner_id, "job_name": stage_name,
                  "protocol": protocol["protocol"]}
        pair = {
            "protocol": protocol["protocol"],
            "jobs": {"short": {"job_id": source_id, "job_name": stage_name},
                     "long": {"job_id": sibling_id, "job_name": sibling_name}},
            "short": {"job_id": candidate_id, "job_name": stage_name},
            "long": {"job_id": sibling_id, "job_name": sibling_name},
            "dependency": "500.pbs101",
        }
        candidate = {"stage": stage, "job_name": stage_name, "job_id": candidate_id,
                     "previous_job_id": previous_job_id, "sequence": 5,
                     "dependency": "500.pbs101", "state": "submitted"}
        summary = {"protocol": protocol["protocol"], "arm": "ours_vqwam",
                   "status": "checkpointed", "pbs_job_id": source_summary_id,
                   "rotation_completed": 8, "completed_updates": 0}
        marker = {"protocol": protocol["protocol"], "status": "claimed",
                  "claimed_job_id": marker_owner,
                  "old_winner": {"stage": stage, "job_id": source_id, "job_name": stage_name},
                  "new_winner": {"stage": stage, "job_id": source_id, "job_name": stage_name}}
        files = {
            root + "/control/ours_race/winner.json": json.dumps(winner).encode(),
            root + "/control/ours_race/submissions_ready.json": json.dumps(pair).encode(),
            root + "/control/ours_race/startup_repair.json": json.dumps(marker).encode(),
            root + "/control/ours_race/startup_repair_claim/owner.json": json.dumps({
                "stage": stage, "job_id": startup_owner,
            }).encode(),
            root + "/control/ours_race/claim": b"",
            root + "/control/" + stage + "_handle.json": json.dumps(candidate).encode(),
            root + "/results/ours/summary.json": json.dumps(summary).encode(),
            root + "/results/ours/checkpoint.pt": b"last committed checkpoint",
            root + "/artifacts/" + candidate_id + "/job.log": (
                log_text if log_text is not None else
                "Startup-repair claim is already held; refusing duplicate retry\n"
            ).encode(),
        }
        if add_control_sources:
            files[root + "/artifacts/" + candidate_id + "/control_sources"] = b"present"
        statuses = {
            source_id: (0, f"Job_Name = {stage_name}\njob_state = F\nExit_status = 0\n"),
            candidate_id: (0, f"Job_Name = {stage_name}\njob_state = {candidate_state}\nExit_status = {candidate_exit}\n"),
            sibling_id: (0, f"Job_Name = {sibling_name}\njob_state = F\n"),
        }
        remote = PreclaimResumeRemote(statuses, files)
        remote.protocol_root = root
        race_dir = Path(directory) / "ours_race"
        race_dir.mkdir(parents=True)
        (race_dir / "submissions_ready.json").write_text(json.dumps(pair), encoding="utf-8")
        (Path(directory) / f"{stage}_handle.json").write_text(json.dumps(candidate), encoding="utf-8")
        return remote, candidate_id, source_id, sibling_id

    def test_ours_preclaim_startup_failure_resume_uses_only_proven_committed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(control, "HERE", Path(directory)):
            remote, candidate_id, source_id, sibling_id = self._ours_preclaim_resume_fixture(directory)
            result = control.resume_ours(remote, object(), self.protocol)

            self.assertEqual(result["resumed_stage"], "ours_short")
            self.assertEqual(result["preclaim_resume_proof"]["kind"],
                             "failed_stale_startup_claim_before_model_load")
            self.assertEqual(result["job"]["previous_job_id"], candidate_id)
            self.assertEqual(result["job"]["sequence"], 6)
            self.assertEqual(result["race_receipt"]["short"]["job_id"], "404.pbs101")
            self.assertEqual(result["race_receipt"]["jobs"]["short"]["job_id"], "404.pbs101")
            self.assertEqual(result["race_receipt"]["long"]["job_id"], sibling_id)
            archive = Path(directory) / "ours_race" / "archive" / (
                "preclaim_resume_" + candidate_id.replace(".", "_") + "_submissions_ready.json")
            self.assertTrue(archive.is_file())
            self.assertEqual(json.loads(archive.read_text(encoding="utf-8"))["short"]["job_id"], candidate_id)
            intent = json.loads((Path(directory) / "ours_short_r6_intent.json").read_text(encoding="utf-8"))
            self.assertEqual(intent["repair_kind"], "ours_preclaim_startup_failure_resume")
            self.assertEqual(intent["repair_proof"]["source_winner_job_id"], source_id)
            self.assertTrue(any(command.startswith("qsub ") for command in remote.commands))
            self.assertFalse(any("rm " in command for command in remote.commands))
            qsub_index = next(i for i, event in enumerate(remote.events) if event[0] == "qsub")
            pair_reads = [i for i, event in enumerate(remote.events)
                          if event[0] == "read_pair" and event[1].endswith("/submissions_ready.json")]
            qrls_index = next(i for i, event in enumerate(remote.events) if event[0] == "qrls")
            qsub_command = remote.events[qsub_index][1]
            self.assertIn(" -h ", qsub_command)
            self.assertTrue(any(qsub_index < index < qrls_index for index in pair_reads))
            self.assertEqual(remote.events[qrls_index], ("qrls", "404.pbs101"))
            self.assertEqual(remote.release_pair_at_qrls["short"]["job_id"], "404.pbs101")
            self.assertEqual(remote.release_receipt_at_qrls["status"], "pending")
            self.assertEqual(result["release_receipt"]["status"], "released")

    def test_ours_resume_uncertain_qrls_keeps_receipt_and_refuses_another_submission(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(control, "HERE", Path(directory)):
            remote, _, _, _ = self._ours_preclaim_resume_fixture(directory)
            remote.qrls_return_code = 1
            with self.assertRaisesRegex(RuntimeError, "qrls outcome.*uncertain.*reconcile"):
                control.resume_ours(remote, object(), self.protocol)
            self.assertEqual(sum(command.startswith("qsub ") for command in remote.commands), 1)
            release_path = Path(directory) / "ours_race" / "resume_release_404_pbs101.json"
            release = json.loads(release_path.read_text(encoding="utf-8"))
            self.assertEqual(release["status"], "uncertain")
            remote_release = json.loads(remote.files[
                self.protocol["remote_root"] + "/control/ours_race/resume_release_404_pbs101.json"])
            self.assertEqual(remote_release, release)
            with self.assertRaisesRegex(RuntimeError, "release outcome is unresolved"):
                control.resume_ours(remote, object(), self.protocol)
            self.assertEqual(sum(command.startswith("qsub ") for command in remote.commands), 1)

    def test_ours_normal_resume_is_held_until_updated_pair_is_read_back(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(control, "HERE", Path(directory)):
            remote, _, source_id, sibling_id = self._ours_preclaim_resume_fixture(
                directory, candidate_job_id="601.pbs101", candidate_exit=0)
            result = control.resume_ours(remote, object(), self.protocol)
            self.assertIsNone(result["preclaim_resume_proof"])
            self.assertEqual(result["race_receipt"]["short"]["job_id"], "404.pbs101")
            self.assertEqual(result["race_receipt"]["long"]["job_id"], sibling_id)
            self.assertEqual(result["release_receipt"]["status"], "released")
            qsub_index = next(i for i, event in enumerate(remote.events) if event[0] == "qsub")
            pair_reads = [i for i, event in enumerate(remote.events)
                          if event[0] == "read_pair" and event[1].endswith("/submissions_ready.json")]
            qrls_index = next(i for i, event in enumerate(remote.events) if event[0] == "qrls")
            self.assertIn(" -h ", remote.events[qsub_index][1])
            self.assertTrue(any(qsub_index < index < qrls_index for index in pair_reads))
            self.assertEqual(remote.events[qrls_index], ("qrls", "404.pbs101"))
            self.assertEqual(remote.release_pair_at_qrls["short"]["job_id"], "404.pbs101")
            self.assertEqual(remote.release_receipt_at_qrls["status"], "pending")

    def test_ours_preclaim_resume_rejects_live_mismatched_owner_source_and_loaded_model(self):
        cases = (
            ("live retry", {"candidate_state": "R"}, "exact F/Exit_status 64"),
            ("wrong lineage", {"previous_job_id": "604.pbs101"}, "exact winner lineage"),
            ("new race owner", {"winner_id": "603.pbs101"}, "same terminal race winner"),
            ("wrong startup marker owner", {"marker_owner": "603.pbs101"}, "belongs to another owner"),
            ("wrong startup lock owner", {"startup_owner": "603.pbs101"}, "belongs to another owner"),
            ("wrong source summary", {"source_summary_id": "603.pbs101"}, "exact committed R8/VQ0"),
            ("loaded model", {"add_control_sources": True}, "reached source snapshot/model setup"),
            ("resume ownership", {"log_text": "RACE_RESUME_WINNER stage=ours_short\\n"},
             "lacks the exact stale-startup-claim error"),
        )
        for label, overrides, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory, \
                    patch.object(control, "HERE", Path(directory)):
                remote, _, _, _ = self._ours_preclaim_resume_fixture(directory, **overrides)
                with self.assertRaisesRegex(RuntimeError, message):
                    control.resume_ours(remote, object(), self.protocol)
                self.assertFalse(any(command.startswith("qsub ") for command in remote.commands))

    def test_job_wrapper_uses_calibration_entrypoint_for_ours_stage(self):
        script = (control.HERE / "job.sh").read_text(encoding="utf-8")
        start = script.index('run_python "$COSMOS_CODE_ROOT/calibrate_ours.py"')
        ours = script[start:].split(";;", 1)[0]
        self.assertIn("calibrate_ours.py", ours)
        self.assertNotIn("train.py", ours)
        self.assertIn("RACE_STARTUP_REPAIR_WINNER", script)

    def test_checkpointed_resume_requires_job_success_summary_and_marker(self):
        stage, job_id = "teacher", "202.pbs101"
        name = control.stages(self.protocol)[stage][0]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(control, "HERE", Path(directory)), patch.object(
                    control, "read_remote_json", return_value={
                        "protocol": self.protocol["protocol"], "status": "checkpointed", "pbs_job_id": job_id,
                    }):
                (Path(directory) / f"{stage}_handle.json").write_text(json.dumps({
                    "stage": stage, "job_name": name, "job_id": job_id, "sequence": 0,
                }), encoding="utf-8")
                remote = FakeRemote({job_id: (0, f"Job_Name = {name}\njob_state = F\nExit_status = 0\n")})
                result = control.submit(remote, object(), self.protocol, stage, resume=True)
                self.assertEqual(result["previous_job_id"], job_id)
                self.assertTrue(any(command.startswith("test -f ") for command in remote.commands))

    def test_checkpointed_resume_rejects_missing_exit_status(self):
        stage, job_id = "teacher", "202.pbs101"
        name = control.stages(self.protocol)[stage][0]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(control, "HERE", Path(directory)):
                (Path(directory) / f"{stage}_handle.json").write_text(json.dumps({
                    "stage": stage, "job_name": name, "job_id": job_id, "sequence": 0,
                }), encoding="utf-8")
                remote = FakeRemote({job_id: (0, f"Job_Name = {name}\njob_state = F\n")})
                with self.assertRaisesRegex(RuntimeError, "Exit_status 0"):
                    control.submit(remote, object(), self.protocol, stage, resume=True)


class QstatAttributeBoundaryTests(unittest.TestCase):
    def test_submit_argument_continuation_is_not_a_live_dependency(self):
        output = ('    job_state = Q\n    Submit_arguments = qsub -W\n'
                  '\t depend=afterok:1.pbs /scratch/script.pbs\n    project = p\n')
        self.assertIsNone(control._qstat_attribute(output, 'depend'))
        self.assertEqual(control._qstat_attribute(output, 'job_state'), 'Q')

    def test_tab_wrapped_dependency_ids_are_rejoined_without_spaces(self):
        output = ('    depend = afterok:1.pbs@pbs101,beforeok:2.pb\n'
                  '\ts@pbs101\n    Submit_arguments = qsub -W depend=old\n')
        self.assertEqual(control._qstat_attribute(output, 'depend'),
                         'afterok:1.pbs@pbs101,beforeok:2.pbs@pbs101')


if __name__ == "__main__":
    unittest.main()
