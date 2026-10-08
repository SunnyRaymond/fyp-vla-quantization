"""Bounded ASPIRE2A control and small-file transfer for this pilot."""
from __future__ import annotations

import argparse
import itertools
import shlex
import socket
import stat
import sys
import time
from pathlib import Path

import paramiko

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT / "nscc-access"))
import aspire2a_shell as access  # noqa: E402

REMOTE = "/scratch/users/ntu/yguo017/fastwam-libero-plus-pilot-20261005"
LIMIT = 256 * 1024


def direct_connect():
    credentials = access.read_credentials()
    sock = socket.create_connection((access.ASPIRE2A_HOST, 22), timeout=20)
    transport = paramiko.Transport(sock)
    transport.channel_timeout = 30
    transport.start_client(timeout=20)
    access.verify_host_key(transport, access.ASPIRE2A_HOST, access.NSCC_KNOWN_HOSTS)
    transport.auth_password(credentials["NSCC_USERNAME"], credentials["NSCC_PASSWORD"])
    if not transport.is_authenticated():
        transport.close()
        raise RuntimeError("Trusted direct authentication failed")
    return None, transport


def connect():
    try:
        pair = access.connect()
        pair[1].channel_timeout = 30
        # Fail over now if the accepted jump route cannot open even a control channel.
        probe = pair[1].open_session(timeout=12)
        probe.exec_command("true")
        probe.recv_exit_status()
        probe.close()
        return pair
    except (OSError, EOFError, paramiko.SSHException) as exc:
        try:
            if "pair" in locals():
                for item in pair:
                    if item:
                        item.close()
        except Exception:
            pass
        print(f"Jump-route session unavailable ({type(exc).__name__}); retrying via trusted direct route.")
        return direct_connect()


def sftp_client(transport):
    channel = transport.open_session(timeout=60)
    channel.settimeout(30)
    try:
        channel.invoke_subsystem("sftp")
        return paramiko.SFTPClient(channel)
    except Exception:
        channel.close()
        raise


def command(transport, value, timeout=90):
    channel = transport.open_session(timeout=60)
    channel.settimeout(timeout)
    channel.exec_command(value)
    out, err = bytearray(), bytearray()
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() >= deadline:
            channel.close()
            raise TimeoutError(f"Remote command exceeded {timeout}s: {value[:120]}")
        received = False
        while channel.recv_ready():
            out.extend(channel.recv(32768))
            received = True
        while channel.recv_stderr_ready():
            err.extend(channel.recv_stderr(32768))
            received = True
        if channel.exit_status_ready():
            break
        if not received:
            time.sleep(0.02)
    while channel.recv_ready():
        out.extend(channel.recv(32768))
    while channel.recv_stderr_ready():
        err.extend(channel.recv_stderr(32768))
    code = channel.recv_exit_status()
    channel.close()
    if err:
        print(err.decode("utf-8", errors="replace").rstrip(), file=sys.stderr)
    return code, out.decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("probe", "upload", "submit", "submit-preflight", "submit-layout", "find-layout", "stat-assets", "mark-assets", "status", "pull"))
    parser.add_argument("--job")
    parser.add_argument("--files", nargs="*")
    args = parser.parse_args()
    jump, transport = connect()
    try:
        if args.action == "probe":
            cmd = (
                "printf 'host='; hostname -s; printf 'user='; id -un; "
                "printf 'jobs\\n'; qstat -u yguo017; "
                "for p in /scratch/users/ntu/yguo017/fastwam-smoke/FastWAM "
                "/scratch/users/ntu/yguo017/fastwam-smoke/venv "
                "/scratch/users/ntu/yguo017/fastwam-smoke/libero "
                "/scratch/users/ntu/yguo017/fastwam-smoke/libero/libero "
                "/scratch/users/ntu/yguo017/fastwam-smoke/libero/libero/benchmark "
                "/scratch/users/ntu/yguo017/fastwam-smoke/libero/libero/benchmark/libero_task_suite_map.py "
                "/scratch/users/ntu/yguo017/fastwam-smoke/libero-config/config.yaml "
                "/scratch/users/ntu/yguo017/fastwam-smoke/libero/libero/assets "
                "/scratch/users/ntu/yguo017/fastwam-smoke/libero/libero/assets/new_objects "
                "/scratch/users/ntu/yguo017/fastwam-smoke/libero-plus/libero/libero/assets "
                "/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/containers/libero-latest.sif "
                "/scratch/users/ntu/yguo017/fastwam-libero-plus-pilot-20261005; do "
                "if test -e \"$p\"; then stat -c '%n %F %s' \"$p\"; else printf 'MISSING %s\\n' \"$p\"; fi; done"
            )
            code, output = command(transport, cmd)
            print(output.rstrip())
            return code
        if args.action == "upload":
            code, output = command(transport, "mkdir -p " + shlex.quote(REMOTE))
            if code:
                raise RuntimeError(output)
            names = args.files or [p.name for p in HERE.iterdir() if p.suffix in {".py", ".pbs", ".md"}]
            with sftp_client(transport) as sftp:
                for name in names:
                    path = HERE / name
                    if path.parent.resolve() != HERE.resolve() or not path.is_file():
                        raise RuntimeError(f"Not a task control file: {name}")
                    data = path.read_bytes()
                    if len(data) > LIMIT:
                        raise RuntimeError(f"Control file exceeds 256 KiB: {name}")
                    with sftp.file(f"{REMOTE}/{name}", "wb") as target:
                        target.write(data)
                    print(f"Uploaded {name}: {len(data)} bytes")
            return 0
        if args.action == "submit":
            code, output = command(transport, "qsub " + shlex.quote(f"{REMOTE}/prepare_env.pbs"))
            print(output.rstrip())
            return code
        if args.action == "submit-preflight":
            if not args.job:
                parser.error("submit-preflight requires --job")
            dependency = shlex.quote(f"afterok:{args.job}")
            target = shlex.quote(f"{REMOTE}/preflight_env.pbs")
            code, output = command(transport, f"qsub -W depend={dependency} {target}")
            print(output.rstrip())
            return code
        if args.action == "submit-layout":
            if not args.job:
                parser.error("submit-layout requires --job")
            dependency = shlex.quote(f"afterany:{args.job}")
            target = shlex.quote(f"{REMOTE}/inspect_assets.pbs")
            code, output = command(transport, f"qsub -W depend={dependency} {target}")
            print(output.rstrip())
            return code
        if args.action == "find-layout":
            code, output = command(transport, "qselect -x -u yguo017 -N libero_plus_layout 2>&1")
            print(output.rstrip())
            return code
        if args.action == "stat-assets":
            plus_root = (
                f"{REMOTE}/LIBERO-plus/libero/libero/inspire/hdd/project/embodied-multimodality/"
                "public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets"
            )
            base_root = "/scratch/users/ntu/yguo017/fastwam-smoke/libero/libero/assets"
            sftp = sftp_client(transport)
            try:
                for label, root in (("PLUS_ASSETS", plus_root), ("BASE_ASSETS", base_root)):
                    try:
                        attrs = list(itertools.islice(sftp.listdir_iter(root, read_aheads=1), 12))
                    except OSError as exc:
                        print(f"{label} MISSING {root} ({type(exc).__name__})")
                        continue
                    print(f"{label} {root}")
                    for attr in attrs[:12]:
                        kind = "DIR" if stat.S_ISDIR(attr.st_mode) else "FILE"
                        print(f"  {kind} {attr.filename} {attr.st_size}")
            finally:
                sftp.close()
            return 0
        if args.action == "mark-assets":
            if not args.job:
                parser.error("mark-assets requires --job")
            if args.job != "25691968.pbs101":
                raise RuntimeError("Refusing to mark assets without the verified extraction job")
            plus_root = (
                f"{REMOTE}/LIBERO-plus/libero/libero/inspire/hdd/project/embodied-multimodality/"
                "public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets"
            )
            log_path = f"{REMOTE}/artifacts/{args.job}/job.log"
            marker_path = f"{REMOTE}/assets-b548dd25ee0401c46217ba7e3614a598e8979e48.complete"
            expected_receipts = (
                "asset_download_complete bytes=6395849578 expected=6395849578",
                "ASSET_EXTRACT_COMPLETE files=457675",
            )
            sftp = sftp_client(transport)
            try:
                with sftp.file(log_path, "rb") as remote_log:
                    log = remote_log.read(LIMIT + 1)
                if len(log) > LIMIT:
                    raise RuntimeError("Extraction receipt log exceeds the 256 KiB control-file limit")
                decoded = log.decode("utf-8", errors="replace")
                for receipt in expected_receipts:
                    if receipt not in decoded:
                        raise RuntimeError(f"Missing verified extraction receipt: {receipt}")
                entries = (
                    ("articulated_objects", True), ("new_objects", True), ("scenes", True),
                    ("stable_hope_objects", True), ("stable_scanned_objects", True),
                    ("textures", True), ("turbosquid_objects", True),
                    ("serving_region.xml", False), ("wall_frames.stl", False), ("wall.xml", False),
                )
                for name, is_dir in entries:
                    attrs = sftp.stat(f"{plus_root}/{name}")
                    if is_dir and not stat.S_ISDIR(attrs.st_mode):
                        raise RuntimeError(f"Expected asset directory at {name}")
                    if not is_dir and not stat.S_ISREG(attrs.st_mode):
                        raise RuntimeError(f"Expected asset file at {name}")
                try:
                    with sftp.file(marker_path, "rb") as marker:
                        existing = marker.read(128).decode("ascii", errors="replace").strip()
                except FileNotFoundError:
                    existing = ""
                commit = "b548dd25ee0401c46217ba7e3614a598e8979e48"
                if existing and existing != commit:
                    raise RuntimeError("Existing asset completion marker has an unexpected commit")
                if not existing:
                    temp_path = marker_path + ".tmp"
                    with sftp.file(temp_path, "wb") as marker:
                        marker.write((commit + "\n").encode("ascii"))
                    sftp.rename(temp_path, marker_path)
                print(f"Verified extraction receipts and 10 expected paths; asset marker ready: {marker_path}")
            finally:
                sftp.close()
            return 0
        if args.action == "status":
            if not args.job:
                code, output = command(transport, "qstat -u yguo017")
            else:
                code, output = command(transport, "qstat -fx " + shlex.quote(args.job))
                candidates = [
                    f"{REMOTE}/LIBERO-plus/libero/libero/assets/articulated_objects",
                    f"{REMOTE}/LIBERO-plus/libero/libero/assets/textures",
                    f"{REMOTE}/LIBERO-plus/libero/libero/assets/assets/articulated_objects",
                    f"{REMOTE}/LIBERO-plus/libero/libero/assets/assets/textures",
                    f"{REMOTE}/LIBERO-plus/libero/libero/articulated_objects",
                    f"{REMOTE}/LIBERO-plus/libero/libero/libero/assets/articulated_objects",
                    f"{REMOTE}/LIBERO-plus/libero/libero/libero/assets/textures",
                    f"{REMOTE}/LIBERO-plus/libero/libero/libero/libero/assets/articulated_objects",
                    f"{REMOTE}/LIBERO-plus/libero/libero/libero/libero/libero/assets/articulated_objects",
                    f"{REMOTE}/LIBERO-plus/libero/libero/LIBERO-plus/libero/libero/assets/articulated_objects",
                    f"{REMOTE}/LIBERO-plus/libero/libero/libero/assets/articulated_objects",
                    f"{REMOTE}/LIBERO-plus/libero/libero/libero/assets/textures",
                    f"{REMOTE}/LIBERO-plus/libero/libero/assets/LIBERO-plus/libero/libero/assets/articulated_objects",
                ]
                candidates.extend(
                    f"/scratch/users/ntu/yguo017/fastwam-smoke/libero/libero/assets/{name}"
                    for name in ("articulated_objects", "new_objects", "scenes", "stable_hope_objects",
                                 "stable_scanned_objects", "textures", "turbosquid_objects")
                )
                plus_asset_root = (
                    f"{REMOTE}/LIBERO-plus/libero/libero/inspire/hdd/project/embodied-multimodality/"
                    "public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets"
                )
                candidates.extend(f"{plus_asset_root}/{name}" for name in (
                    "articulated_objects", "new_objects", "scenes", "stable_hope_objects",
                    "stable_scanned_objects", "textures", "turbosquid_objects",
                    "serving_region.xml", "wall_frames.stl", "wall.xml",
                ))
                candidates.extend(f"/scratch/users/ntu/yguo017/fastwam-smoke/libero/libero/assets/{name}"
                                  for name in ("serving_region.xml", "wall_frames.stl", "wall.xml"))
                candidate_loop = "for p in " + " ".join(map(shlex.quote, candidates)) + "; do if test -e \"$p\"; then stat -c 'PRESENT %F %n' \"$p\"; else printf 'MISSING %s\\n' \"$p\"; fi; done"
                tail_cmd = (
                    "test ! -f " + shlex.quote(f"{REMOTE}/artifacts/{args.job}/job.log")
                    + " || tail -c 5000 " + shlex.quote(f"{REMOTE}/artifacts/{args.job}/job.log")
                    + "; printf '\\n--- fixed asset candidate dirs ---\\n'; " + candidate_loop
                )
                _, tail = command(transport, tail_cmd)
                output += "\n--- job log tail ---\n" + tail
            print(output.rstrip())
            return code
        if not args.job:
            parser.error("pull requires --job")
        names = args.files or ["env_status.json", "manifest_summary.json", "job.log", "qstat.txt"]
        dest = HERE / "artifacts" / args.job
        dest.mkdir(parents=True, exist_ok=True)
        with sftp_client(transport) as sftp:
            for name in names:
                remote = f"{REMOTE}/{name}" if name in {"env_status.json", "manifest_summary.json"} else f"{REMOTE}/artifacts/{args.job}/{name}"
                try:
                    size = sftp.stat(remote).st_size
                except FileNotFoundError:
                    continue
                if size > LIMIT:
                    print(f"Skipped {name}: {size} bytes exceeds 256 KiB")
                    continue
                sftp.get(remote, str(dest / name))
                print(f"Retrieved {name}: {size} bytes")
        return 0
    finally:
        transport.close()
        if jump:
            jump.close()


if __name__ == "__main__":
    raise SystemExit(main())
