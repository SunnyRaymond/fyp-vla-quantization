"""Local controller of the EEE standard-LIBERO 840 loop. Run from the project root with the nscc-access venv:

  control.py upload                      code -> /projects/yguo017/laq840/code
  control.py submit KEY -- SBATCH_ARGS   sbatch on the login node, job id saved under KEY in jobs.json
  control.py status                      queue, episode counts, FAILED markers, last job-log lines
  control.py pull                        summaries, startup checks, episodes -> pulled/
"""
from __future__ import annotations

import json
import posixpath
import sys
from pathlib import Path
import shlex

import paramiko

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "eee-access"))
import eee_shell as eee  # noqa: E402

REMOTE = "/projects/yguo017/laq840"
CODE = ["l840.py", "fw840.py", "cos840.py", "test840.py", "analysis840.py", "stack840.sh", "slots840.sh",
        "run840.sbatch", "cputest.sbatch", "unpack.sbatch", "probe.sbatch", "protocol.json", "t5prep840.py",
        "t5keys.sbatch"]
JOBS = HERE / "jobs.json"
MODELS = ("fastwam", "cosmos")


def run(transport, command, timeout=120):
    session = transport.open_session(timeout=20)
    session.settimeout(timeout)
    session.exec_command("bash -lc " + eee.quote(command))
    out = session.makefile("rb").read().decode("utf-8", "replace")
    err = session.makefile_stderr("rb").read().decode("utf-8", "replace")
    return session.recv_exit_status(), out, err


def jobs():
    return json.loads(JOBS.read_text(encoding="utf-8")) if JOBS.exists() else {}


def upload(t):
    sftp = paramiko.SFTPClient.from_transport(t)
    for d in (REMOTE, f"{REMOTE}/code", f"{REMOTE}/logs"):
        try:
            sftp.stat(d)
        except FileNotFoundError:
            sftp.mkdir(d)
    for name in CODE:
        target = f"{REMOTE}/code/{name}"
        sftp.put(str(HERE / name), target + ".tmp")
        sftp.posix_rename(target + ".tmp", target)
    print("UPLOADED", len(CODE), "files")


def submit(t, key, args):
    rc, out, err = run(t, f"cd {REMOTE}/code && sbatch --parsable " + " ".join(shlex.quote(a) for a in args))
    if rc != 0:
        raise SystemExit(f"sbatch failed: {err or out}")
    job = out.strip().split(";")[0]
    state = jobs()
    state[key] = dict(job=job, args=args)
    JOBS.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print("SUBMITTED", key, job)


def status(t):
    ids = ",".join(v["job"] for v in jobs().values())
    lines = ["squeue -u $USER -o '%.10i %.9P %.10j %.2t %.10M %.10l %.6D %R %q %b'"]
    if ids:
        lines.append(f"sacct -X -n -j {ids} -o JobID,JobName%10,State%14,Elapsed,ExitCode,NodeList%18")
    for m in MODELS:
        lines.append(f"echo {m}: episodes $(ls {REMOTE}/results/{m}/episodes 2>/dev/null | wc -l) "
                     f"claims $(ls {REMOTE}/results/{m}/claims 2>/dev/null | wc -l)")
        lines.append(f"for d in {REMOTE}/artifacts/{m}/*/; do [ -d \"$d\" ] || continue; echo \"== $d\"; "
                     f"cat \"$d/FAILED\" 2>/dev/null; cat \"$d/exit_code.txt\" 2>/dev/null; "
                     f"grep -h -E 'STARTUP|SLOT_|EPISODE|FAILED|Error|error' \"$d\"/slot_*.log 2>/dev/null | tail -n 4; "
                     f"grep -E 'GPU_SAMPLE' \"$d/job.log\" 2>/dev/null | tail -n 2; done")
    rc, out, err = run(t, "; ".join(lines))
    print(out, err)


def pull(t):
    sftp = paramiko.SFTPClient.from_transport(t)
    dest = HERE / "pulled"

    def get(remote, local):
        sftp.stat(remote)  # raises FileNotFoundError before paramiko leaves an empty local file behind
        local.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote, str(local))

    count = 0
    for m in MODELS:
        base = f"{REMOTE}/results/{m}"
        for name in ("summary.json", "RESULTS.zh.md", "manifest.json"):
            try:
                get(f"{base}/{name}", dest / "results" / m / name)
            except FileNotFoundError:
                pass
        for sub in ("episodes",):
            try:
                names = sftp.listdir(f"{base}/{sub}")
            except FileNotFoundError:
                continue
            for name in names:
                if name.endswith(".json"):
                    get(f"{base}/{sub}/{name}", dest / "results" / m / sub / name)
                    count += 1
        try:
            for w in sftp.listdir(f"{base}/workers"):
                try:
                    get(f"{base}/workers/{w}/startup.json", dest / "results" / m / "workers" / w / "startup.json")
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            pass
        try:
            for a in sftp.listdir(f"{REMOTE}/artifacts/{m}"):
                for name in ("job.log", "exit_code.txt", "FAILED", "PIPELINE_COMPLETE"):
                    try:
                        get(f"{REMOTE}/artifacts/{m}/{a}/{name}", dest / "artifacts" / m / a / name)
                    except FileNotFoundError:
                        pass
        except FileNotFoundError:
            pass
    print("PULLED episodes", count)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    t = eee.connect()
    try:
        cmd = sys.argv[1]
        if cmd == "upload":
            upload(t)
        elif cmd == "submit":
            key, rest = sys.argv[2], sys.argv[3:]
            if rest[:1] == ["--"]:
                rest = rest[1:]
            submit(t, key, rest)
        elif cmd == "status":
            status(t)
        elif cmd == "pull":
            pull(t)
        else:
            raise SystemExit(__doc__)
    finally:
        t.close()


if __name__ == "__main__":
    main()
