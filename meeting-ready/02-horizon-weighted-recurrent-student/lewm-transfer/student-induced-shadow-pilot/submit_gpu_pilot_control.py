"""Upload small frozen control files and submit the guarded GPU PBS job."""

from __future__ import annotations

import argparse
import importlib.util
import json
import socket
from pathlib import Path


HERE = Path(__file__).resolve().parent
ACCESS = HERE.parents[3] / "nscc-access" / "aspire2a_shell.py"
REMOTE = "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/student-induced-shadow-pilot"
SELECTION_MANIFEST_REMOTE = "/scratch/users/ntu/yguo017/lewm-pusht-iteration/artifacts/student-induced-shadow-pilot/real-observation-training-data/selection/25543404.pbs101/selection_manifest.json"
REAL_OBS_CAPTURE_REMOTE = "/scratch/users/ntu/yguo017/lewm-pusht-iteration/artifacts/student-induced-shadow-pilot/real-observation-training-data/capture/25543623.pbs101"
COMMON_FILES = (
    "PILOT_FREEZE.json",
    "run_student_induced_shadow_pilot.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--diagnostic", action="store_true")
    mode.add_argument("--control-control", action="store_true")
    mode.add_argument("--seeded-control-control", action="store_true")
    mode.add_argument("--seeded-pilot", action="store_true")
    mode.add_argument("--posthoc-absolute-regret", action="store_true")
    mode.add_argument("--fullbank-ranker", action="store_true")
    mode.add_argument("--oracle-shortlist-posthoc", action="store_true")
    mode.add_argument("--real-observation-probe", action="store_true")
    mode.add_argument("--real-observation-probe-14", action="store_true")
    mode.add_argument("--real-observation-training-data-select", action="store_true")
    mode.add_argument("--real-observation-training-data-capture", action="store_true")
    mode.add_argument("--real-observation-finetune", action="store_true")
    mode.add_argument("--real-observation-recover", action="store_true")
    mode.add_argument("--proposal-dispersion-posthoc", action="store_true")
    mode.add_argument("--action-error-coupling-posthoc", action="store_true")
    mode.add_argument("--partial-horizon-teacher-hybrid", action="store_true")
    mode.add_argument("--partial-horizon-independent-gate", action="store_true")
    mode.add_argument("--latent-score-sensitivity-posthoc", action="store_true")
    parser.add_argument("--direct", action="store_true", help="Use verified direct ASPIRE2A SSH when NTU jump host is unavailable")
    args = parser.parse_args()
    if args.real_observation_training_data_select:
        freeze = json.loads((HERE / "real-observation-training-data" / "FREEZE.json").read_text(encoding="utf-8"))
        if freeze.get("episode_selection", {}).get("selection_manifest_job_id"):
            parser.error("the frozen 80-task selection is already materialized; do not submit another selector")
    if args.real_observation_probe_14:
        wrapper = "real-observation-rollout-probe-14-task/run_real_observation_rollout_probe_14_task.pbs"
        files = COMMON_FILES + ("SEEDED_PILOT_FREEZE.json",) + tuple(
            f"real-observation-rollout-probe-14-task/{name}" for name in (
                "FREEZE.json", "run_real_observation_rollout_probe_14_task.py",
                "run_real_observation_rollout_probe_14_task.pbs",
            )
        )
    elif args.real_observation_training_data_select:
        wrapper = "real-observation-training-data/select_real_observation_training_episodes_cpu.pbs"
        files = (
            "SEEDED_PILOT_FREEZE.json",
            "SELECTION_FREEZE.json",
            "onpolicy-fullbank-ranker/FREEZE.json",
            "onpolicy-fullbank-ranker/results/25538135.pbs101/collection_manifest.json",
            "real-observation-training-data/FREEZE.json",
            "real-observation-training-data/README.zh.md",
            "real-observation-training-data/select_real_observation_training_episodes.py",
            "real-observation-training-data/select_real_observation_training_episodes_cpu.pbs",
        )
    elif args.real_observation_training_data_capture:
        wrapper = "real-observation-training-data/collect_real_observation_training_data.pbs"
        files = COMMON_FILES + ("SEEDED_PILOT_FREEZE.json",) + tuple(
            f"real-observation-training-data/{name}" for name in (
                "FREEZE.json", "README.zh.md", "collect_real_observation_training_data.py",
                "collect_real_observation_training_data.pbs",
            )
        ) + ("real-observation-rollout-probe-14-task/run_real_observation_rollout_probe_14_task.py",)
    elif args.real_observation_finetune:
        wrapper = "real-observation-finetune/run_real_observation_finetune.pbs"
        files = tuple(f"real-observation-finetune/{name}" for name in (
            "FREEZE.json", "run_real_observation_finetune.py", "run_real_observation_finetune.pbs",
        ))
    elif args.real_observation_recover:
        wrapper = "real-observation-training-data/recover_real_observation_training_data_cpu.pbs"
        files = tuple(f"real-observation-training-data/{name}" for name in (
            "recover_real_observation_training_data.py", "recover_real_observation_training_data_cpu.pbs",
        ))
    elif args.proposal_dispersion_posthoc:
        wrapper = "proposal-dispersion-posthoc/run_proposal_dispersion_cpu.pbs"
        files = tuple(f"proposal-dispersion-posthoc/{name}" for name in (
            "FREEZE.json", "diagnose_proposal_dispersion.py", "run_proposal_dispersion_cpu.pbs",
        ))
    elif args.action_error_coupling_posthoc:
        wrapper = "action-error-coupling-posthoc/run_action_error_coupling_cpu.pbs"
        files = tuple(f"action-error-coupling-posthoc/{name}" for name in (
            "FREEZE.json", "diagnose_action_error_coupling.py", "run_action_error_coupling_cpu.pbs",
        ))
    elif args.partial_horizon_teacher_hybrid:
        wrapper = "onpolicy-fullbank-ranker/partial-horizon-teacher-hybrid/run_partial_horizon_teacher_hybrid.pbs"
        files = tuple(f"onpolicy-fullbank-ranker/partial-horizon-teacher-hybrid/{name}" for name in (
            "FREEZE.json", "runner.py", "run_partial_horizon_teacher_hybrid.pbs",
        ))
    elif args.partial_horizon_independent_gate:
        wrapper = "partial-horizon-independent-gate/run_partial_horizon_independent_gate.pbs"
        files = tuple(f"partial-horizon-independent-gate/{name}" for name in (
            "FREEZE.json", "runner.py", "run_partial_horizon_independent_gate.pbs",
        ))
    elif args.latent_score_sensitivity_posthoc:
        wrapper = "latent-score-sensitivity-posthoc/run_latent_score_sensitivity.pbs"
        files = tuple(f"latent-score-sensitivity-posthoc/{name}" for name in (
            "FREEZE.json", "runner.py", "run_latent_score_sensitivity.pbs",
        ))
    elif args.real_observation_probe:
        wrapper = "real-observation-rollout-probe/run_real_observation_rollout_probe.pbs"
        files = COMMON_FILES + ("SEEDED_PILOT_FREEZE.json",) + tuple(
            f"real-observation-rollout-probe/{name}" for name in (
                "FREEZE.json", "run_real_observation_rollout_probe.py",
                "run_real_observation_rollout_probe.pbs",
            )
        )
    elif args.oracle_shortlist_posthoc:
        wrapper = "onpolicy-fullbank-ranker/run_fullbank_oracle_shortlist_cpu.pbs"
        files = tuple(f"onpolicy-fullbank-ranker/{name}" for name in (
            "diagnose_fullbank_oracle_shortlist.py", "run_fullbank_oracle_shortlist_cpu.pbs",
        ))
    elif args.fullbank_ranker:
        wrapper = "onpolicy-fullbank-ranker/run_fullbank_ranker.pbs"
        files = COMMON_FILES + ("SEEDED_PILOT_FREEZE.json",) + tuple(
            f"onpolicy-fullbank-ranker/{name}" for name in (
                "FREEZE.json", "collect_fullbank.py", "train_fullbank_residual_ranker.py",
                "run_fullbank_ranker.pbs",
            )
        )
    elif args.posthoc_absolute_regret:
        wrapper = "run_posthoc_absolute_regret_analysis.pbs"
        files = ("posthoc_absolute_regret_analysis.py", wrapper)
    elif args.seeded_pilot:
        wrapper = "run_student_induced_shadow_pilot_seeded_variant.pbs"
        extra = ("SEEDED_PILOT_FREEZE.json",)
    elif args.seeded_control_control:
        wrapper = "run_student_induced_shadow_seeded_control_control_repro.pbs"
        extra = ("SEEDED_CONTROL_CONTROL_REPRO_FREEZE.json",)
    elif args.control_control:
        wrapper = "run_student_induced_shadow_control_control_repro.pbs"
        extra = ("CONTROL_CONTROL_REPRO_FREEZE.json",)
    elif args.diagnostic:
        wrapper = "run_student_induced_shadow_diagnostic.pbs"
        extra = ("DIAGNOSTIC_TWO_PAIR_FREEZE.json",)
    else:
        wrapper = "run_student_induced_shadow_pilot.pbs"
        extra = ()
    if not any((args.posthoc_absolute_regret, args.fullbank_ranker,
                args.oracle_shortlist_posthoc, args.real_observation_probe,
                args.real_observation_probe_14, args.real_observation_training_data_select,
                args.real_observation_training_data_capture, args.real_observation_finetune,
                args.real_observation_recover, args.proposal_dispersion_posthoc,
                args.action_error_coupling_posthoc, args.partial_horizon_teacher_hybrid,
                args.partial_horizon_independent_gate, args.latent_score_sensitivity_posthoc)):
        files = COMMON_FILES + extra + (wrapper,)
    spec = importlib.util.spec_from_file_location("aspire2a_shell", ACCESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load verified ASPIRE2A connector")
    connector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(connector)
    if args.direct:
        credentials = connector.read_credentials()
        sock = socket.create_connection((connector.ASPIRE2A_HOST, 22), timeout=20)
        nscc = connector.paramiko.Transport(sock)
        try:
            nscc.start_client(timeout=20)
            connector.verify_host_key(nscc, connector.ASPIRE2A_HOST, connector.NSCC_KNOWN_HOSTS)
            nscc.auth_password(credentials["NSCC_USERNAME"], credentials["NSCC_PASSWORD"])
            if not nscc.is_authenticated():
                raise RuntimeError("verified direct ASPIRE2A authentication failed")
        except BaseException:
            nscc.close()
            raise
        jump = None
    else:
        jump, nscc = connector.connect()
    try:
        sftp = nscc.open_sftp_client()
        try:
            sftp.stat(REMOTE)
            if args.fullbank_ranker or args.oracle_shortlist_posthoc:
                child = f"{REMOTE}/onpolicy-fullbank-ranker"
                try:
                    sftp.stat(child)
                except OSError:
                    sftp.mkdir(child)
            if args.real_observation_probe:
                child = f"{REMOTE}/real-observation-rollout-probe"
                try:
                    sftp.stat(child)
                except OSError:
                    sftp.mkdir(child)
            if args.real_observation_probe_14:
                child = f"{REMOTE}/real-observation-rollout-probe-14-task"
                try:
                    sftp.stat(child)
                except OSError:
                    sftp.mkdir(child)
            if args.real_observation_training_data_select:
                child = f"{REMOTE}/real-observation-training-data"
                try:
                    sftp.stat(child)
                except OSError:
                    sftp.mkdir(child)
            if args.real_observation_training_data_capture:
                for child in (
                    f"{REMOTE}/real-observation-training-data",
                    f"{REMOTE}/real-observation-rollout-probe-14-task",
                ):
                    try:
                        sftp.stat(child)
                    except OSError:
                        sftp.mkdir(child)
            if args.real_observation_finetune:
                child = f"{REMOTE}/real-observation-finetune"
                try:
                    sftp.stat(child)
                except OSError:
                    sftp.mkdir(child)
            if args.real_observation_recover:
                child = f"{REMOTE}/real-observation-training-data"
                try:
                    sftp.stat(child)
                except OSError:
                    sftp.mkdir(child)
            if args.proposal_dispersion_posthoc:
                child = f"{REMOTE}/proposal-dispersion-posthoc"
                try:
                    sftp.stat(child)
                except OSError:
                    sftp.mkdir(child)
            if args.action_error_coupling_posthoc:
                child = f"{REMOTE}/action-error-coupling-posthoc"
                try:
                    sftp.stat(child)
                except OSError:
                    sftp.mkdir(child)
            if args.partial_horizon_teacher_hybrid:
                child = f"{REMOTE}/onpolicy-fullbank-ranker/partial-horizon-teacher-hybrid"
                try:
                    sftp.stat(child)
                except OSError:
                    sftp.mkdir(child)
            if args.partial_horizon_independent_gate:
                child = f"{REMOTE}/partial-horizon-independent-gate"
                try:
                    sftp.stat(child)
                except OSError:
                    sftp.mkdir(child)
            if args.latent_score_sensitivity_posthoc:
                child = f"{REMOTE}/latent-score-sensitivity-posthoc"
                try:
                    sftp.stat(child)
                except OSError:
                    sftp.mkdir(child)
            for name in files:
                source = HERE / name
                if not source.is_file() or source.stat().st_size > 200_000:
                    raise RuntimeError(f"missing or unexpectedly large control file: {name}")
                destination = f"{REMOTE}/{name}"
                sftp.put(str(source), destination)
                if sftp.stat(destination).st_size != source.stat().st_size:
                    raise RuntimeError(f"control-file upload size mismatch: {name}")
                print(f"uploaded {name} ({source.stat().st_size} bytes)")
            if args.real_observation_training_data_select:
                required_remote_sources = (
                    f"{REMOTE}/onpolicy-fullbank-ranker/results/25538135.pbs101/collection_manifest.json",
                    "/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/official-lewm-dataset-teacher-baseline/artifacts/25534994.pbs101/selected_tasks.json",
                )
                for source_path in required_remote_sources:
                    source_stat = sftp.stat(source_path)
                    if source_stat.st_size > 200_000:
                        raise RuntimeError(f"expected small frozen selection input: {source_path}")
            if args.real_observation_training_data_capture:
                for source_path in (
                    SELECTION_MANIFEST_REMOTE,
                    f"{REMOTE.rsplit('/', 1)[0]}/interface-probe/artifacts/24554356.pbs101/interface_probe.json",
                ):
                    source_stat = sftp.stat(source_path)
                    if source_stat.st_size > 200_000:
                        raise RuntimeError(f"expected small frozen capture input: {source_path}")
            if args.real_observation_finetune:
                summary_path = f"{REAL_OBS_CAPTURE_REMOTE}/collection_summary_recovered.json"
                archive_path = f"{REAL_OBS_CAPTURE_REMOTE}/real_observation_training_data.npz"
                summary_stat = sftp.stat(summary_path)
                if summary_stat.st_size > 200_000:
                    raise RuntimeError("capture summary is unexpectedly large")
                with sftp.open(summary_path, "r") as stream:
                    summary = json.loads(stream.read().decode("utf-8"))
                if not str(summary.get("status", "")).startswith("COMPLETE_") or len(summary.get("episodes", [])) != 80 or int(summary.get("sample_count", 0)) != 141 or summary.get("recovery_provenance", {}).get("source_pbs_exit_status") != 1:
                    raise RuntimeError("independent capture recovery is not complete with 80 episodes and 141 samples")
                if sftp.stat(archive_path).st_size <= 0:
                    raise RuntimeError("capture archive is empty")
                sftp.stat(f"{REMOTE.rsplit('/', 1)[0]}/run_lewm_recurrent_student.py")
            if args.real_observation_recover:
                for source_path in (
                    f"{REAL_OBS_CAPTURE_REMOTE}/collection_summary.json",
                    f"{REAL_OBS_CAPTURE_REMOTE}/real_observation_training_data.npz",
                    SELECTION_MANIFEST_REMOTE,
                ):
                    sftp.stat(source_path)
            if args.proposal_dispersion_posthoc:
                bank_root = f"{REMOTE}/onpolicy-fullbank-ranker/artifacts/25538135.pbs101"
                sftp.stat(f"{bank_root}/train_banks.pt")
                sftp.stat(f"{bank_root}/validation_banks.pt")
            if args.action_error_coupling_posthoc:
                bank_root = f"{REMOTE}/onpolicy-fullbank-ranker/artifacts/25538135.pbs101"
                sftp.stat(f"{bank_root}/train_banks.pt")
                stage_root = "/scratch/users/ntu/yguo017/lewm-pusht-iteration/artifacts/student-induced-shadow-pilot"
                sftp.stat(f"{stage_root}/real-observation-rollout-probe/25542215.pbs101/real_observation_latent_alignment.json")
                sftp.stat(f"{stage_root}/real-observation-rollout-probe-14-task/25542467.pbs101/real_observation_latent_alignment_14_task.json")
            if args.partial_horizon_teacher_hybrid:
                sftp.stat(f"{REMOTE}/onpolicy-fullbank-ranker/artifacts/25538135.pbs101/train_banks.pt")
                sftp.stat(f"{REMOTE.rsplit('/', 1)[0]}/cem-distribution-distill/artifacts/25239551.pbs101/treatment_step1000.pt")
                sftp.stat("/scratch/users/ntu/yguo017/lewm-pusht-iteration/stablewm_home/pusht/lewm_object.ckpt")
            if args.partial_horizon_independent_gate:
                sftp.stat(f"{REMOTE}/onpolicy-fullbank-ranker/collect_fullbank.py")
                sftp.stat(f"{REMOTE}/onpolicy-fullbank-ranker/partial-horizon-teacher-hybrid/runner.py")
                sftp.stat(f"{REMOTE.rsplit('/', 1)[0]}/cem-distribution-distill/artifacts/25239551.pbs101/treatment_step1000.pt")
                sftp.stat("/scratch/users/ntu/yguo017/lewm-pusht-iteration/stablewm_home/pusht/lewm_object.ckpt")
            if args.latent_score_sensitivity_posthoc:
                sftp.stat(f"{REMOTE}/onpolicy-fullbank-ranker/artifacts/25538135.pbs101/train_banks.pt")
                sftp.stat(f"{REMOTE}/onpolicy-fullbank-ranker/partial-horizon-teacher-hybrid/runner.py")
                sftp.stat(f"{REMOTE.rsplit('/', 1)[0]}/cem-distribution-distill/artifacts/25239551.pbs101/treatment_step1000.pt")
                sftp.stat("/scratch/users/ntu/yguo017/lewm-pusht-iteration/stablewm_home/pusht/lewm_object.ckpt")
        finally:
            sftp.close()

        channel = nscc.open_session(timeout=20)
        try:
            if args.real_observation_training_data_capture:
                command = f"qsub -v LEWM_REAL_OBS_SELECTION_MANIFEST={SELECTION_MANIFEST_REMOTE} {REMOTE}/{wrapper}"
            else:
                command = f"qsub {REMOTE}/{wrapper}"
            channel.exec_command(command)
            stdout = channel.makefile("rb").read().decode("utf-8", errors="replace").strip()
            stderr = channel.makefile_stderr("rb").read().decode("utf-8", errors="replace").strip()
            status = channel.recv_exit_status()
            if status != 0:
                raise RuntimeError(f"qsub failed (exit {status}): {stderr or stdout}")
            print(f"submitted {stdout}")
        finally:
            channel.close()
    finally:
        nscc.close()
        if jump is not None:
            jump.close()


if __name__ == "__main__":
    main()
