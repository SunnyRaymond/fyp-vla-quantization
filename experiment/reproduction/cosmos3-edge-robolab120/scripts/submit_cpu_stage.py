from __future__ import annotations

import argparse
import posixpath
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "nscc-access"))

import aspire2a_shell  # noqa: E402
import paramiko  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent
LOCAL_LOG_DIR = SCRIPTS_DIR.parent / "logs"
CONTROLS = ("prepare_cpu_stage.pbs", "install_runtime_ubuntu22.sh", "run_one_episode.sh", "run_policy_server.sh")
CHANNEL_TIMEOUT = 60
COMMAND_TIMEOUT = 90


def run_remote(transport: paramiko.Transport, command: str) -> tuple[int, str, str]:
    channel = transport.open_session(timeout=CHANNEL_TIMEOUT)
    try:
        channel.settimeout(CHANNEL_TIMEOUT)
        channel.exec_command(command)
        out = bytearray()
        err = bytearray()
        deadline = time.monotonic() + COMMAND_TIMEOUT
        while True:
            while channel.recv_ready():
                out.extend(channel.recv(65536))
            while channel.recv_stderr_ready():
                err.extend(channel.recv_stderr(65536))
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                return channel.recv_exit_status(), out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"remote command exceeded {COMMAND_TIMEOUT}s")
            time.sleep(0.1)
    finally:
        channel.close()


def mkdirs(sftp: paramiko.SFTPClient, path: str) -> None:
    parts = path.strip("/").split("/")
    current = ""
    for part in parts:
        current += "/" + part
        try:
            sftp.mkdir(current)
        except OSError:
            sftp.stat(current)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-existing", action="store_true", help="qstat only; do not upload or submit")
    parser.add_argument("--upload-support-files", action="store_true", help="upload the README and one-episode launcher only")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--status", metavar="JOB_ID")
    parser.add_argument("--cancel", metavar="JOB_ID", help="qdel a job already confirmed as this task and running")
    args = parser.parse_args()
    if sum((args.check_existing, args.upload_support_files, args.submit, bool(args.status), bool(args.cancel))) != 1:
        parser.error("choose exactly one operation: --check-existing, --upload-support-files, --submit, --status JOB_ID, or --cancel JOB_ID")

    LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    job_marker = LOCAL_LOG_DIR / "cpu_stage_job_id.txt"
    intent_marker = LOCAL_LOG_DIR / "cpu_stage_submit_intent.txt"
    submit_record = LOCAL_LOG_DIR / "cpu_stage_submit.txt"
    if args.submit and (job_marker.exists() or intent_marker.exists()):
        raise SystemExit(f"Refusing duplicate or unresolved submission marker: {job_marker if job_marker.exists() else intent_marker}")

    jump, nscc = aspire2a_shell.connect()
    try:
        if args.status:
            scratch_log = "/scratch/users/ntu/yguo017/cosmos3-edge-robolab120/logs/job.log"
            job_marker = shlex.quote("PBS_JOBID=" + args.status)
            error_pattern = shlex.quote("error:|FAIL:|Traceback|No solution")
            status_command = (
                "qstat -xf " + shlex.quote(args.status)
                + "; status_rc=$?; printf '\\n--- matching job error context ---\\n'; "
                + "start=$(grep -n -F -- " + job_marker + " " + shlex.quote(scratch_log)
                + " | tail -n 1 | cut -d: -f1); "
                + "if [ -n \"$start\" ]; then tail -n +\"$start\" " + shlex.quote(scratch_log)
                + " | grep -i -m1 -B5 -A30 -E " + error_pattern
                + " || tail -n 20 " + shlex.quote(scratch_log)
                + "; else tail -n 20 " + shlex.quote(scratch_log) + "; fi; exit $status_rc"
            )
            rc, out, err = run_remote(nscc, status_command)
            snapshot = LOCAL_LOG_DIR / "cpu_stage_latest_status.txt"
            snapshot.write_text(
                out + ("\n--- stderr ---\n" + err if err else ""), encoding="utf-8"
            )
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            print(out, end="")
            if err:
                print(err, file=sys.stderr, end="")
            return rc
        if args.cancel:
            rc, out, err = run_remote(nscc, "qdel " + shlex.quote(args.cancel))
            if out:
                print(out, end="")
            if err:
                print(err, file=sys.stderr, end="")
            return rc

        rc, existing_jobs, stderr = run_remote(nscc, "qstat -u yguo017")
        if stderr:
            print(stderr, file=sys.stderr, end="")
        if rc != 0:
            print("Could not confirm existing PBS state; no upload or submission was attempted.", file=sys.stderr)
            return rc
        matches = [line for line in existing_jobs.splitlines() if "cos3edge_stage" in line]
        print("QSTAT_USER_JOBS:")
        print(existing_jobs, end="" if existing_jobs.endswith("\n") else "\n")
        if matches:
            print("EXISTING_COSMOS_STAGE_JOB:")
            print("\n".join(matches))
            if args.submit:
                return 2
        if args.check_existing:
            return 0

        if args.upload_support_files:
            sftp = paramiko.SFTPClient.from_transport(nscc)
            try:
                sftp.get_channel().settimeout(CHANNEL_TIMEOUT)
                remote_home = sftp.normalize(".")
                task_root = posixpath.join(remote_home, "experiment", "reproduction", "cosmos3-edge-robolab120")
                scripts_dir = posixpath.join(task_root, "scripts")
                mkdirs(sftp, scripts_dir)
                sftp.put(str(SCRIPTS_DIR.parent / "README.md"), posixpath.join(task_root, "README.md"))
                for name in ("run_one_episode.sh", "run_policy_server.sh"):
                    launcher = posixpath.join(scripts_dir, name)
                    sftp.put(str(SCRIPTS_DIR / name), launcher)
                    sftp.chmod(launcher, 0o700)
            finally:
                sftp.close()
            print(f"REMOTE_README={posixpath.join(task_root, 'README.md')}")
            print(f"REMOTE_EPISODE_LAUNCHER={posixpath.join(scripts_dir, 'run_one_episode.sh')}")
            print(f"REMOTE_POLICY_SERVER_LAUNCHER={posixpath.join(scripts_dir, 'run_policy_server.sh')}")
            return 0

        if matches:
            return 2

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        intent_marker.write_text(f"created_utc={stamp}\nstate=unresolved_until_qsub_receipt\n", encoding="utf-8")

        sftp = paramiko.SFTPClient.from_transport(nscc)
        try:
            sftp.get_channel().settimeout(CHANNEL_TIMEOUT)
            remote_home = sftp.normalize(".")
            task_root = posixpath.join(remote_home, "experiment", "reproduction", "cosmos3-edge-robolab120")
            scripts_dir = posixpath.join(task_root, "scripts")
            logs_dir = posixpath.join(task_root, "logs")
            # The task tree already exists from prior staging attempts. Avoid
            # walking/mkdir-ing shared home ancestors over the latency-prone SFTP channel.
            for name in CONTROLS:
                remote_file = posixpath.join(scripts_dir, name)
                sftp.put(str(SCRIPTS_DIR / name), remote_file)
                sftp.chmod(remote_file, 0o700)
        finally:
            sftp.close()

        remote_log = posixpath.join(logs_dir, f"cpu-stage-{stamp}.log")
        receipt = posixpath.join(logs_dir, "cpu_stage_job_id.txt")
        script = posixpath.join(scripts_dir, "prepare_cpu_stage.pbs")
        qsub = (
            "cd "
            + shlex.quote(scripts_dir)
            + " && jobid=$(qsub -N cos3edge_stage -o "
            + shlex.quote(remote_log)
            + " "
            + shlex.quote(script)
            + ") || exit $?; tmp="
            + shlex.quote(receipt + ".tmp")
            + "; printf '%s\\n' \"$jobid\" > \"$tmp\" && mv -f \"$tmp\" "
            + shlex.quote(receipt)
            + " && printf '%s\\n' \"$jobid\""
        )
        rc, stdout, stderr = run_remote(nscc, qsub)
        if stdout:
            print(stdout, end="")
        if stderr:
            print(stderr, file=sys.stderr, end="")
        if rc != 0:
            print("Submission result is unresolved; reconcile with qstat and remote receipt before retrying.", file=sys.stderr)
            return rc
        job_id = stdout.strip().splitlines()[-1].strip() if stdout.strip() else ""
        if not job_id:
            print("qsub returned no job ID; intent marker retained to block blind resubmission.", file=sys.stderr)
            return 1
        job_marker.write_text(job_id + "\n", encoding="utf-8")
        submit_record.write_text(
            f"job_id={job_id}\nremote_task_root=/scratch/users/ntu/yguo017/cosmos3-edge-robolab120\n"
            f"remote_log={remote_log}\nremote_receipt={receipt}\n",
            encoding="utf-8",
        )
        intent_marker.unlink(missing_ok=True)
        print(f"JOB_ID={job_id}")
        print(f"REMOTE_LOG={remote_log}")
        print(f"REMOTE_TASK_ROOT=/scratch/users/ntu/yguo017/cosmos3-edge-robolab120")
        return 0
    finally:
        nscc.close()
        jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
