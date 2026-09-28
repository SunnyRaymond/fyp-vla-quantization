"""Bounded metadata/source preparation; run only in a real PBS allocation."""
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import urllib.request


def allocation_guard():
    job = os.environ.get("PBS_JOBID", "")
    nodefile = Path(os.environ.get("PBS_NODEFILE", "/nonexistent"))
    host = socket.gethostname().split(".")[0].lower()
    if not job or not nodefile.is_file() or any(x in host for x in ("login", "head", "submit")):
        raise RuntimeError("A genuine PBS compute allocation is required")
    nodes = {x.split(".")[0].lower() for x in nodefile.read_text().split()}
    if host not in nodes:
        raise RuntimeError("Current host is absent from PBS_NODEFILE")
    return job, host


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "rolling-ball-lewm-preflight"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(8_000_001)
    if len(data) > 8_000_000:
        raise RuntimeError("Metadata/source response exceeds bounded preparation limit")
    return data


def get_json(url):
    return json.loads(fetch(url))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    job, host = allocation_guard()
    root = Path("/scratch/users/ntu/yguo017/lewm-rolling-ball-async")
    out = root / "runs" / job
    out.mkdir(parents=True, exist_ok=True)
    report = {"job_id": job, "host": host, "status": "RUNNING", "trained": False,
              "closed_loop_evaluated": False, "scope": "Rolling Ball Interception only"}
    write_json(out / "preparation.json", report)
    try:
        repo = "https://api.github.com/repos/LxRoboticsLab/ReflexBench"
        revision = "8bb931485093c6d98f8729774ad01bf824964e16"
        tree = get_json(repo + "/git/trees/" + revision + "?recursive=1")
        if tree.get("truncated"):
            raise RuntimeError("ReflexBench source tree listing was truncated")
        selected = [x["path"] for x in tree["tree"] if x["type"] == "blob" and (
            "/rolling_ball_interception/" in x["path"] or x["path"] in (
                "README.md", "source/reflexbench/setup.py", "source/reflexbench/pyproject.toml",
                "scripts/evaluation/eval.py", "scripts/evaluation/policy_client.py",
                "scripts/evaluation/POLICY_SERVER.md", "scripts/collect_vla_data.py"))]
        if not selected or len(selected) > 60:
            raise RuntimeError("Unexpected bounded task source selection")
        for relative in selected:
            target = out / "reflexbench_source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(fetch("https://raw.githubusercontent.com/LxRoboticsLab/ReflexBench/"
                                    + revision + "/" + relative))
        report["reflexbench_commit"] = revision
        report["source_files"] = selected
        hf = "https://huggingface.co/api/datasets/cyx337/ReflexBench_dataset"
        dataset = {"sha": "9295b6e9878609a992047f0b8b65421a493299e7"}
        report["dataset_commit"] = dataset["sha"]
        root_entries = get_json(hf + "/tree/" + dataset["sha"])
        rolling = [x for x in root_entries if "rolling" in x["path"].lower()]
        write_json(out / "dataset_root.json", root_entries)
        report["rolling_entries"] = rolling
        if len(rolling) == 1 and rolling[0]["type"] == "directory":
            directory = rolling[0]["path"]
            report["dataset_layout"] = "task_directory"
        elif {x["path"] for x in root_entries if x["type"] == "directory"} >= {"data", "meta", "videos"}:
            directory = "meta"
            report["dataset_layout"] = "joint_lerobot_dataset"
        else:
            raise RuntimeError("Dataset layout requires explicit resolution")
        entries = get_json(hf + "/tree/" + dataset["sha"] + "/" + directory + "?recursive=true")
        write_json(out / "rolling_files.json", entries)
        # Metadata only: no videos, parquet, model weights, or other task datasets.
        metadata = [x for x in entries if x["type"] == "file" and (
                    x["path"].endswith(("meta/info.json", "meta/stats.json", "README.md")) or
                    (directory == "meta" and x["path"].endswith((".parquet", ".jsonl"))))]
        for entry in metadata:
            relative = entry["path"]
            target = out / "dataset_metadata" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(fetch("https://huggingface.co/datasets/cyx337/ReflexBench_dataset/resolve/"
                                    + dataset["sha"] + "/" + relative))
        report["metadata_files"] = [x["path"] for x in metadata]
        report["listed_metadata_bytes"] = sum(x.get("size", 0) for x in metadata)
        packages = {}
        for name in ("torch", "stable-worldmodel", "isaacsim", "isaaclab", "lerobot", "pyarrow", "av"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        report["existing_lewm_environment_packages"] = packages
        if directory == "meta":
            import pyarrow.parquet as pq
            task_path = out / "dataset_metadata" / "meta" / "tasks.parquet"
            tasks = pq.read_table(task_path).to_pylist()
            write_json(out / "dataset_tasks.json", tasks)
            report["tasks"] = tasks
            rolling_tasks = [x for x in tasks if "rolling" in json.dumps(x).lower()]
            if not rolling_tasks:
                raise RuntimeError("No Rolling Ball task label found in combined dataset")
            labels = {v for row in rolling_tasks for v in row.values() if isinstance(v, str)}
            episodes = []
            for path in (out / "dataset_metadata" / "meta" / "episodes").rglob("*.parquet"):
                for row in pq.read_table(path).to_pylist():
                    row_tasks = row.get("tasks", [])
                    if isinstance(row_tasks, str):
                        row_tasks = [row_tasks]
                    if set(row_tasks) & labels:
                        episodes.append({k: v for k, v in row.items() if not k.startswith("stats/")})
            write_json(out / "rolling_episodes.json", episodes)
            report["rolling_episode_count"] = len(episodes)
            if not episodes:
                raise RuntimeError("Rolling Ball task found but episode mapping is unresolved")
        versions = {}
        for name in ("le-wm", "stable-worldmodel"):
            source = Path("/scratch/users/ntu/yguo017/lewm-pusht-iteration") / name
            git_dir = source / ".git"
            head = git_dir / "HEAD"
            if not head.is_file():
                versions[name] = {"path": str(source), "revision": None,
                                  "note": "No standalone Git HEAD; identity remains unverified"}
                continue
            reference = head.read_text().strip()
            if reference.startswith("ref: "):
                reference = reference[5:]
                loose = git_dir / reference
                revision = loose.read_text().strip() if loose.is_file() else None
                packed = git_dir / "packed-refs"
                if revision is None and packed.is_file():
                    for line in packed.read_text().splitlines():
                        fields = line.split()
                        if len(fields) == 2 and fields[1] == reference:
                            revision = fields[0]
                            break
            else:
                revision = reference
            versions[name] = {"path": str(source), "revision": revision}
        report["existing_sources"] = versions
        report["status"] = "PASS"
        report["meaning"] = "Pinned source and dataset metadata only; simulation hardware unresolved"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        write_json(out / "preparation.json", report)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
