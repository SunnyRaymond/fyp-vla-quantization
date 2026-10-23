#!/usr/bin/env python3
"""Freeze original-LIBERO demonstration snapshots for the calibration cohorts.

Dataset inventory, HDF5 reads, and simulator rendering require an allocated PBS
compute node. ``--control-test`` exercises only the deterministic sampler.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import json
import os
import posixpath
import random
import re
import socket
import ssl
import sys
import time
from urllib.parse import quote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET


PROTOCOL = "fastwam-vq-calibration-suites-v1"
SELECTION_SEED = 20261013
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PHASES = ("early", "middle", "late")
PHASE_PAIRS = ([("early", "middle")] * 4 + [("middle", "late")] * 3
               + [("early", "late")] * 3)
REQUIRED_OBSERVATIONS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
DEFAULT_BASE = Path("/scratch/users/ntu/yguo017/fastwam-smoke")
DEFAULT_HF_REPO = "yifengzhu-hf/LIBERO-datasets"
HTTP_BLOCK_SIZE = 1 << 20
ROW_IDENTITY_KEYS = ("suite", "task_id", "task_name", "domain", "phase", "demo_id", "frame_index",
                     "dimension", "variant", "variant_id", "original_task", "source_index",
                     "environment_seed")


def allocation_guard() -> None:
    """Require the actual host to belong to a live PBS compute allocation."""
    job_id, nodefile = os.environ.get("PBS_JOBID"), os.environ.get("PBS_NODEFILE")
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("Dataset access and rendering require PBS_JOBID and PBS_NODEFILE")
    host = socket.gethostname().split(".", 1)[0]
    allocated = {line.split(".", 1)[0] for line in Path(nodefile).read_text().split()}
    if "login" in host.lower() or host not in allocated:
        raise RuntimeError(f"PBS compute host required; host={host}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise RuntimeError("Calibration input preparation must run in a CPU-only allocation")


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
    os.replace(temporary, path)


def row_identity(row: dict) -> dict:
    return {key: row[key] for key in ROW_IDENTITY_KEYS}


def safe_network_reason(exc: BaseException) -> str:
    """Describe common network failures without exposing URLs or proxy credentials."""
    status = getattr(exc, "code", None)
    if isinstance(status, int):
        return f"HTTP_STATUS_{status}"
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, ssl.SSLCertVerificationError):
        return "TLS_CERTIFICATE_VERIFICATION_FAILED"
    if isinstance(reason, ssl.SSLError):
        return "TLS_ERROR"
    if isinstance(reason, socket.gaierror):
        code = reason.errno
        return f"DNS_RESOLUTION_FAILED errno={code if code is not None else 'unknown'}"
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return "CONNECTION_TIMEOUT"
    if isinstance(reason, ConnectionRefusedError):
        return "CONNECTION_REFUSED"
    if isinstance(reason, OSError):
        code = errno.errorcode.get(reason.errno, "OS_ERROR") if reason.errno is not None else "OS_ERROR"
        return f"{code} errno={reason.errno if reason.errno is not None else 'unknown'}"
    if isinstance(reason, str):
        lowered = reason.lower()
        for needles, label in (
            (("certificate verify", "local issuer"), "TLS_CERTIFICATE_VERIFICATION_FAILED"),
            (("temporary failure in name resolution", "name or service not known",
              "nodename nor servname", "getaddrinfo failed"), "DNS_RESOLUTION_FAILED"),
            (("proxy", "tunnel connection failed"), "PROXY_CONNECTION_FAILED"),
            (("timed out", "timeout"), "CONNECTION_TIMEOUT"),
            (("network is unreachable", "no route to host"), "NETWORK_UNREACHABLE"),
            (("connection refused",), "CONNECTION_REFUSED"),
        ):
            if any(needle in lowered for needle in needles):
                return label
    return f"NETWORK_REASON_CLASS_{type(reason).__name__}"


def source_provenance(row: dict) -> dict:
    return {key: row.get(key) for key in (
        "dataset_source", "dataset_file", "source_repo_id", "source_revision", "source_repo_path",
        "dataset_size_bytes", "demo_length")}


def canonical_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def phase_frame(length: int, phase: str, positions: dict[str, float]) -> int:
    if length < 3 or phase not in PHASES or phase not in positions:
        raise RuntimeError(f"Need at least three states and a declared phase position; length={length}, phase={phase}")
    position = float(positions[phase])
    if not 0 <= position <= 1:
        raise RuntimeError(f"Phase position must be within [0,1]: {phase}={position}")
    return int(round((length - 1) * position))


def freeze_inputs(tasks: list[dict], seed: int = SELECTION_SEED,
                  calibration_seed_base: int = 920_000_000,
                  test_seed_base: int = 930_000_000,
                  phase_pairs: list[tuple[str, str]] | None = None,
                  test_phases: list[str] | None = None,
                  phase_positions: dict[str, float] | None = None,
                  calibration_case_id_start: int = 2000,
                  test_case_id_start: int = 3000) -> tuple[list[dict], dict[str, list[int]]]:
    """Freeze 80 calibration rows, nested C8 IDs, and 40 disjoint-demo tests."""
    if seed != SELECTION_SEED:
        raise RuntimeError(f"The calibration selection seed is frozen at {SELECTION_SEED}")
    by_suite: dict[str, list[dict]] = {suite: [] for suite in SUITES}
    for task in tasks:
        suite = task.get("suite")
        if suite not in by_suite:
            raise RuntimeError(f"Unsupported dataset suite: {suite}")
        by_suite[suite].append(task)
    for suite in SUITES:
        rows = sorted(by_suite[suite], key=lambda task: int(task["task_id"]))
        if len(rows) != 10 or [int(row["task_id"]) for row in rows] != list(range(10)):
            raise RuntimeError(f"{suite} must contain its ten official task IDs 0..9")
        for row in rows:
            demos = row.get("demos", {})
            if len(demos) < 3:
                raise RuntimeError(f"Need three distinct demonstrations: {suite} task {row['task_id']}")
            if any(int(length) < 3 for length in demos.values()):
                raise RuntimeError(f"Every demonstration needs at least three states: {suite} task {row['task_id']}")

    phase_pairs = list(phase_pairs or PHASE_PAIRS)
    test_phases = list(test_phases or (["early"] * 3 + ["middle"] * 4 + ["late"] * 3))
    phase_positions = phase_positions or {"early": 0.2, "middle": 0.5, "late": 0.8}
    if (len(phase_pairs) != 10 or any(tuple(pair) not in PHASE_PAIRS for pair in phase_pairs)
            or [sum(tuple(pair) == candidate for pair in phase_pairs)
                for candidate in (("early", "middle"), ("middle", "late"), ("early", "late"))] != [4, 3, 3]):
        raise RuntimeError("Each suite must use four early/middle, three middle/late, and three early/late task pairs")
    if len(test_phases) != 10 or sorted(test_phases) != sorted(["early"] * 3 + ["middle"] * 4 + ["late"] * 3):
        raise RuntimeError("Each suite must use the frozen test phase allocation E3/M4/L3")
    rng = random.Random(seed)
    inputs: list[dict] = []
    calibration_by_task: dict[tuple[str, int], list[dict]] = {}
    selected_tasks: dict[str, list[int]] = {}
    calibration_rows, test_rows = [], []
    calibration_case_id, test_case_id = calibration_case_id_start, test_case_id_start

    for suite in SUITES:
        suite_tasks = sorted(by_suite[suite], key=lambda task: int(task["task_id"]))
        task_ids = [int(task["task_id"]) for task in suite_tasks]
        task_order = list(task_ids)
        rng.shuffle(task_order)
        phase_pair_by_task = dict(zip(task_order, phase_pairs))
        test_phase_by_task = dict(zip(task_order, test_phases))
        c8_task_ids = sorted(rng.sample(task_ids, 2))
        selected_tasks[suite] = c8_task_ids
        for task in suite_tasks:
            task_id = int(task["task_id"])
            pair, test_phase = phase_pair_by_task[task_id], test_phase_by_task[task_id]
            demo_ids = sorted(int(demo_id) for demo_id in task["demos"])
            chosen_demos = rng.sample(demo_ids, 3)
            task_name = str(task["task_name"])
            demo_lengths = {int(k): int(v) for k, v in task["demos"].items()}
            task_calibration_rows = []
            for demo_id, phase in zip(chosen_demos[:2], pair):
                frame_index = phase_frame(demo_lengths[demo_id], phase, phase_positions)
                row = {
                    "case_id": calibration_case_id,
                    "suite": suite,
                    "task_id": task_id,
                    "task_name": task_name,
                    "domain": "original",
                    "phase": phase,
                    "demo_id": demo_id,
                    "frame_index": frame_index,
                    "sampler_seeds": [calibration_seed_base + calibration_case_id * 2 + i for i in range(2)],
                    "observation_file": f"inputs/case_{calibration_case_id:04d}.npz",
                    "metadata_file": f"inputs/case_{calibration_case_id:04d}.json",
                    "dataset_file": str(task["dataset_file"]),
                    "dataset_source": task["dataset_source"],
                    "source_repo_id": task.get("source_repo_id"),
                    "source_revision": task.get("source_revision"),
                    "source_repo_path": task.get("source_repo_path"),
                    "dataset_size_bytes": task.get("dataset_size_bytes"),
                    "demo_length": demo_lengths[demo_id],
                    "dimension": "original", "variant": "original", "variant_id": None,
                    "original_task": task_name, "source_index": None,
                    "environment_seed": 830_000_000 + calibration_case_id * 17,
                }
                calibration_rows.append(row)
                task_calibration_rows.append(row)
                calibration_case_id += 1
            calibration_by_task[suite, task_id] = task_calibration_rows

            # The test context always uses a third demonstration from this task.
            test_demo = chosen_demos[2]
            test_frame = phase_frame(demo_lengths[test_demo], test_phase, phase_positions)
            test_rows.append({
                "case_id": test_case_id,
                "suite": suite,
                "task_id": task_id,
                "task_name": task_name,
                "domain": "original",
                "phase": test_phase,
                "demo_id": test_demo,
                "frame_index": test_frame,
                "sampler_seeds": [test_seed_base + test_case_id * 2 + i for i in range(2)],
                "observation_file": f"inputs/case_{test_case_id:04d}.npz",
                "metadata_file": f"inputs/case_{test_case_id:04d}.json",
                "dataset_file": str(task["dataset_file"]),
                "dataset_source": task["dataset_source"],
                "source_repo_id": task.get("source_repo_id"),
                "source_revision": task.get("source_revision"),
                "source_repo_path": task.get("source_repo_path"),
                "dataset_size_bytes": task.get("dataset_size_bytes"),
                "demo_length": demo_lengths[test_demo],
                "dimension": "original", "variant": "original", "variant_id": None,
                "original_task": task_name, "source_index": None,
                "environment_seed": 830_000_000 + test_case_id * 17,
            })
            test_case_id += 1

    calibration_rows.sort(key=lambda row: row["case_id"])
    test_rows.sort(key=lambda row: row["case_id"])
    inputs = calibration_rows + test_rows
    c8_ids = []
    for suite in SUITES:
        for task_id in selected_tasks[suite]:
            # Pick one of that task's two calibration snapshots; hence C8 is nested in C80.
            c8_ids.append(rng.choice(calibration_by_task[suite, task_id])["case_id"])
    if (len(calibration_rows) != 80 or [r["case_id"] for r in calibration_rows] != list(range(2000, 2080))
            or len(test_rows) != 40 or [r["case_id"] for r in test_rows] != list(range(3000, 3040))):
        raise RuntimeError("Frozen input IDs/counts do not match the calibration-suite protocol")
    if len(c8_ids) != 8 or len(set(c8_ids)) != 8 or not set(c8_ids) <= {r["case_id"] for r in calibration_rows}:
        raise RuntimeError("C8 must be eight distinct C80 snapshots")
    for suite in SUITES:
        suite_cal = [row for row in calibration_rows if row["suite"] == suite]
        suite_test = [row for row in test_rows if row["suite"] == suite]
        if len(suite_cal) != 20 or len(suite_test) != 10:
            raise RuntimeError(f"Unexpected per-suite counts for {suite}")
        if [sum(row["phase"] == phase for row in suite_cal) for phase in PHASES] != [7, 7, 6]:
            raise RuntimeError(f"Unexpected C80 phase allocation for {suite}")
        if len({row["task_id"] for row in suite_cal}) != 10:
            raise RuntimeError(f"C80 task coverage is incomplete for {suite}")
        if len({row["task_id"] for row in suite_test}) != 10:
            raise RuntimeError(f"Held-out task coverage is incomplete for {suite}")
        if len([row for row in c8_ids if next(r["suite"] for r in calibration_rows if r["case_id"] == row) == suite]) != 2:
            raise RuntimeError(f"C8 must contain two contexts from {suite}")
    for row in test_rows:
        matching = [item for item in calibration_rows
                    if item["suite"] == row["suite"] and item["task_id"] == row["task_id"]]
        if row["demo_id"] in {item["demo_id"] for item in matching}:
            raise RuntimeError(f"Held-out demo overlaps calibration: {row['suite']} task {row['task_id']}")
    cohorts = {"c8": sorted(c8_ids), "c80": [row["case_id"] for row in calibration_rows]}
    return inputs, cohorts


def control_test() -> None:
    tasks = []
    for suite in SUITES:
        for task_id in range(10):
            tasks.append({"suite": suite, "task_id": task_id, "task_name": f"task {suite} {task_id}",
                          "dataset_file": f"/data/{suite}/{task_id}.hdf5",
                          "dataset_source": "local",
                          "demos": {demo: 303 + demo for demo in range(12)}})
    first = freeze_inputs(tasks)
    second = freeze_inputs(tasks)
    assert first == second
    inputs, cohorts = first
    calibration = [row for row in inputs if 2000 <= row["case_id"] < 2080]
    tests = [row for row in inputs if 3000 <= row["case_id"] < 3040]
    assert len(calibration) == 80 and len(tests) == 40
    assert cohorts["c8"] == sorted(cohorts["c8"]) and set(cohorts["c8"]) <= set(cohorts["c80"])
    assert len(cohorts["c8"]) == 8 and len(cohorts["c80"]) == 80
    for suite in SUITES:
        selected = [row for row in calibration if row["suite"] == suite and row["case_id"] in cohorts["c8"]]
        assert len(selected) == 2 and len({row["task_id"] for row in selected}) == 2
        rows = [row for row in calibration if row["suite"] == suite]
        assert [sum(row["phase"] == phase for row in rows) for phase in PHASES] == [7, 7, 6]
        for phase, expected in zip(PHASES, (7, 7, 6)):
            assert sum(row["phase"] == phase for row in rows) == expected
    for row in tests:
        demos = {item["demo_id"] for item in calibration
                 if item["suite"] == row["suite"] and item["task_id"] == row["task_id"]}
        assert row["demo_id"] not in demos
    assert all(row["domain"] == "original" for row in inputs)
    assert all(len(row["sampler_seeds"]) == 2 for row in inputs)
    assert phase_frame(301, "early", {"early": 0.2, "middle": 0.5, "late": 0.8}) == 60
    assert phase_frame(301, "middle", {"early": 0.2, "middle": 0.5, "late": 0.8}) == 150
    assert phase_frame(301, "late", {"early": 0.2, "middle": 0.5, "late": 0.8}) == 240
    fixture = '''<mujoco model="frozen scene">
      <compiler meshdir="/Users/publisher/chiliocosm/assets/stable_scanned_objects"/>
      <asset><mesh name="bowl" file="/Users/publisher/chiliocosm/assets/stable_scanned_objects/../objects/../stable_scanned_objects/akita_black_bowl/visual/akita_black_bowl_vis.msh"/>
        <mesh name="reentry" file="/Users/publisher/chiliocosm/assets/../assets/stable_scanned_objects/reentry/mesh.msh"/>
        <mesh name="robosuite" file="/old/robosuite/assets/objects/fixture.stl"/>
        <texture name="relative" file="textures/floor.png"/></asset>
      <worldbody><body name="fixture" pos="0.1 0.2 0.3" quat="1 0 0 0"/></worldbody>
      <camera name="agentview" pos="1 2 3" quat="1 0 0 0"/>
    </mujoco>'''
    rewritten, relocated = relocate_libero_asset_paths(fixture, Path("/original/libero/assets"))
    parsed = ET.fromstring(rewritten)
    files = {node.attrib["name"]: node.attrib["file"] for node in parsed.findall("./asset/mesh")}
    assert parsed.find("./compiler").attrib["meshdir"] == str(Path("/original/libero/assets").resolve() /
                                                               "stable_scanned_objects")
    assert files["bowl"] == str(Path("/original/libero/assets").resolve() / "stable_scanned_objects" /
                                 "akita_black_bowl/visual/akita_black_bowl_vis.msh")
    assert files["robosuite"] == "/old/robosuite/assets/objects/fixture.stl"
    assert files["reentry"] == str(Path("/original/libero/assets").resolve() /
                                   "stable_scanned_objects/reentry/mesh.msh")
    assert parsed.find("./asset/texture").attrib["file"] == "textures/floor.png"
    assert parsed.find("./worldbody/body").attrib["pos"] == "0.1 0.2 0.3"
    assert parsed.find("./camera").attrib["quat"] == "1 0 0 0"
    assert relocated == [("stable_scanned_objects", True),
                         ("stable_scanned_objects/akita_black_bowl/visual/akita_black_bowl_vis.msh", False),
                         ("stable_scanned_objects/reentry/mesh.msh", False)]
    outside_fixture = '<mujoco><asset><mesh file="/Users/publisher/chiliocosm/assets/../outside/mesh.msh"/></asset></mujoco>'
    try:
        relocate_libero_asset_paths(outside_fixture, Path("/original/libero/assets"))
    except RuntimeError as exc:
        assert "tag='mesh'" in str(exc) and "attribute='file'" in str(exc)
        assert "normalized_relative='../outside/mesh.msh'" in str(exc)
    else:
        raise AssertionError("XML path escaping the original LIBERO assets root was accepted")
    print("CONTROL_TEST_OK inputs=80+40 nested_c8=8 phases=7/7/6 xml_asset_relocation=true")


def relocate_libero_asset_paths(model_xml: str, assets_root: Path) -> tuple[str, list[tuple[str, bool]]]:
    """Relocate only absolute paths under the old LIBERO assets root."""
    root = ET.fromstring(model_xml)
    assets_root = Path(assets_root).resolve()
    marker = "/chiliocosm/assets"
    relocated: list[tuple[str, bool]] = []
    for element in root.iter():
        for attribute, original in list(element.attrib.items()):
            normalized = original.replace("\\", "/")
            marker_at = normalized.lower().rfind(marker)
            if marker_at < 0:
                continue
            suffix_at = marker_at + len(marker)
            if suffix_at < len(normalized) and normalized[suffix_at] != "/":
                continue
            prefix = normalized[:marker_at]
            if not (prefix.startswith("/") or re.match(r"^[A-Za-z]:/", prefix)):
                continue
            legacy_root = posixpath.normpath(normalized[:suffix_at])
            normalized_path = posixpath.normpath(normalized)
            try:
                relative_path = PurePosixPath(normalized_path).relative_to(PurePosixPath(legacy_root))
                normalized_relative = relative_path.as_posix()
            except ValueError:
                normalized_relative = posixpath.relpath(normalized_path, legacy_root)
                raise RuntimeError("Demo XML asset path escapes its root: "
                                   f"tag={str(element.tag)!r:.200} attribute={attribute!r:.200} "
                                   f"normalized_relative={normalized_relative!r:.200}")
            parts = relative_path.parts
            target = assets_root.joinpath(*parts)
            element.set(attribute, str(target))
            relocated.append((normalized_relative, attribute.lower() in ("meshdir", "texturedir", "assetdir")))
    if not relocated:
        return model_xml, []
    return ET.tostring(root, encoding="unicode"), relocated


def load_roots(protocol: dict) -> tuple[Path, Path, Path, Path, list[Path]]:
    roots = protocol.get("roots", {})
    data = protocol.get("data", {})
    base = Path(roots.get("base", os.environ.get("BASE", DEFAULT_BASE))).resolve()
    # Accept BASE as either the Fast-WAM checkout root or its original-LIBERO subdirectory.
    if (base / "FastWAM").is_dir():
        fastwam_root = base
    elif (base.parent / "FastWAM").is_dir():
        fastwam_root = base.parent
    else:
        fastwam_root = base
    original_repo = Path(data.get("original_repo", base / "libero")).resolve()
    source_candidates = [original_repo / "libero", base / "libero/libero", base / "libero",
                         fastwam_root / "libero/libero"]
    source_root = next((path.resolve() for path in source_candidates if (path / "benchmark").is_dir()),
                       source_candidates[0].resolve())
    libero_root = source_root.parent
    datasets_value = (data.get("dataset_roots") or roots.get("datasets") or roots.get("libero_datasets")
                      or roots.get("dataset_root") or protocol.get("dataset_root"))
    if isinstance(datasets_value, (str, Path)):
        datasets_value = [datasets_value]
    datasets = ([Path(value).resolve() for value in datasets_value] if datasets_value
                else [(libero_root / "datasets").resolve()])
    compat_value = roots.get("libero_compat") or roots.get("compat")
    compat = Path(compat_value).resolve() if compat_value else (fastwam_root / "libero-compat").resolve()
    config_value = roots.get("libero_config") or roots.get("libero_config_path")
    config = Path(config_value).resolve() if config_value else (fastwam_root / "libero-config").resolve()
    if not config.is_dir():
        raise FileNotFoundError(f"Original LIBERO config directory is unavailable: {config}")
    for path in (source_root, compat, fastwam_root / "FastWAM"):
        if not path.exists():
            raise FileNotFoundError(f"Original LIBERO render dependency is unavailable: {path}")
    return fastwam_root, source_root, compat, config, datasets


def dataset_candidates(dataset_roots: list[Path], suite: str, task_name: str) -> tuple[Path | None, list[str]]:
    wanted = canonical_name(task_name)
    searched, found = [], []
    for dataset_root in dataset_roots:
        suite_root = dataset_root / suite
        searched.append(str(suite_root / f"{wanted}_demo.hdf5"))
        if not suite_root.is_dir():
            continue
        matches = []
        for path in (*suite_root.glob("*.hdf5"), *suite_root.glob("*.h5")):
            stem = canonical_name(path.stem)
            if stem.endswith("_demo"):
                stem = stem[:-5]
            if stem == wanted:
                matches.append(path.resolve())
        if len(matches) > 1:
            raise RuntimeError(f"Ambiguous original HDF5 for {suite}/{task_name!r}: {matches}")
        if matches:
            # dataset_roots are an ordered fallback list; keep the first configured copy.
            return matches[0], searched
    return None, searched


def api_json(url: str, purpose: str):
    """Read a public HF API response without auth and without surfacing redirect URLs."""
    request = Request(url, headers={"User-Agent": "fastwam-vq-calibration-input-preparer/1.0"})
    try:
        with urlopen(request, timeout=45) as response:
            payload = response.read(16 << 20)
            if response.read(1):
                raise RuntimeError(f"Hugging Face {purpose} response exceeds the 16 MiB control limit")
        return json.loads(payload.decode("utf-8"))
    except RuntimeError:
        raise
    except Exception as exc:
        reason = safe_network_reason(exc)
        raise RuntimeError(f"Could not read public Hugging Face {purpose} API: "
                           f"error_class={type(exc).__name__} reason={reason}") from None


def hf_repo_path(repo_id: str, revision: str, repo_path: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
        raise RuntimeError("Invalid source_hf_repo_id in the frozen protocol")
    parts = repo_path.split("/")
    if not repo_path or repo_path.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise RuntimeError("Hugging Face API returned an unsafe dataset path")
    return f"https://huggingface.co/datasets/{quote(repo_id, safe='/')}/resolve/{quote(revision, safe='')}/{quote(repo_path, safe='/')}"


def load_source_receipt(out: Path, repo_id: str, create: bool = False) -> dict | None:
    receipt_path = out / "source_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("repo_id") != repo_id or not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("revision", ""))):
            raise RuntimeError("Existing source_receipt.json does not match the requested public HF repository/revision")
        receipt.setdefault("range_verified_paths", [])
        receipt.setdefault("range_unsupported_paths", [])
        receipt.setdefault("downloaded_paths", [])
        return receipt
    if not create:
        return None
    endpoint = f"https://huggingface.co/api/datasets/{quote(repo_id, safe='/')}/revision/main"
    response = api_json(endpoint, "revision")
    revision = str(response.get("sha", response.get("commit", ""))).lower()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Public Hugging Face revision API did not return a 40-character commit SHA")
    receipt = {
        "source": "public Hugging Face dataset repository",
        "repo_id": repo_id,
        "revision_ref": "main",
        "revision": revision,
        "revision_api": "https://huggingface.co/api/datasets/<repo>/revision/main",
        "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
        "authentication": "none",
        "range_verified_paths": [],
        "range_unsupported_paths": [],
        "downloaded_paths": [],
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def hf_tree_files(repo_id: str, revision: str) -> list[dict]:
    base = f"https://huggingface.co/api/datasets/{quote(repo_id, safe='/')}/tree/{quote(revision, safe='')}"
    url = base + "?" + urlencode({"recursive": "true", "expand": "false", "limit": "1000"})
    files, seen_urls = [], set()
    for _ in range(20):
        if url in seen_urls:
            raise RuntimeError("Hugging Face tree API returned a repeated pagination cursor")
        seen_urls.add(url)
        request = Request(url, headers={"User-Agent": "fastwam-vq-calibration-input-preparer/1.0"})
        try:
            with urlopen(request, timeout=45) as response:
                payload = response.read(16 << 20)
                if response.read(1):
                    raise RuntimeError("Hugging Face tree API page exceeds the 16 MiB control limit")
                link_header = response.headers.get("Link", "")
        except RuntimeError:
            raise
        except Exception as exc:
            reason = safe_network_reason(exc)
            raise RuntimeError("Could not list the pinned public Hugging Face dataset tree: "
                               f"error_class={type(exc).__name__} reason={reason}") from None
        page = json.loads(payload.decode("utf-8"))
        if not isinstance(page, list):
            raise RuntimeError("Hugging Face tree API returned an unexpected response")
        files.extend(item for item in page if item.get("type") == "file")
        match = re.search(r'<([^>]+)>\s*;\s*rel="?next"?', link_header)
        if not match:
            return files
        next_url = urljoin(url, match.group(1))
        parsed = urlparse(next_url)
        if parsed.scheme != "https" or parsed.netloc != "huggingface.co" or not parsed.path.startswith("/api/datasets/"):
            raise RuntimeError("Hugging Face tree API returned an unsafe pagination link")
        url = next_url
    raise RuntimeError("Hugging Face tree API exceeded the 20-page control limit")


def find_hf_dataset_file(files: list[dict], suite: str, task_name: str) -> dict | None:
    wanted, suite_key = canonical_name(task_name), canonical_name(suite)
    matches = []
    for item in files:
        repo_path = str(item.get("path", ""))
        if Path(repo_path).suffix.lower() not in (".h5", ".hdf5"):
            continue
        parts = repo_path.split("/")
        if suite_key not in {canonical_name(part) for part in parts[:-1]}:
            continue
        stem = canonical_name(Path(repo_path).stem)
        if stem.endswith("_demo"):
            stem = stem[:-5]
        if stem == wanted:
            matches.append(item)
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous original HDF5 in pinned public source: {suite}/{task_name!r}")
    return matches[0] if matches else None


def remote_source(repo_id: str, revision: str, item: dict) -> dict:
    repo_path = str(item["path"])
    return {"dataset_source": "huggingface", "dataset_file": f"hf://{repo_id}@{revision}/{repo_path}",
            "source_repo_id": repo_id, "source_revision": revision, "source_repo_path": repo_path,
            "dataset_size_bytes": int(item["size"]) if item.get("size") is not None else None}


def safe_cache_file(out: Path, repo_id: str, revision: str, repo_path: str) -> Path:
    relative = Path(*repo_path.split("/"))
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise RuntimeError("Unsafe path cannot be used in the CPU source cache")
    cache_root = (out / "source_cache" / repo_id / revision).resolve()
    target = (cache_root / relative).resolve()
    target.relative_to(cache_root)
    return target


def probe_hf_ranges(remote: dict) -> bool:
    url = hf_repo_path(remote["source_repo_id"], remote["source_revision"], remote["source_repo_path"])
    request = Request(url, headers={"Range": "bytes=0-0",
                                    "User-Agent": "fastwam-vq-calibration-input-preparer/1.0"})
    try:
        with urlopen(request, timeout=60) as response:
            status = int(response.status)
            content_range = response.headers.get("Content-Range", "")
            sample = response.read(1) if status == 206 else b""
    except Exception as exc:
        reason = safe_network_reason(exc)
        raise RuntimeError("Could not verify public Hugging Face byte-range support: "
                           f"error_class={type(exc).__name__} reason={reason}") from None
    return status == 206 and bool(re.fullmatch(r"bytes 0-0/[1-9][0-9]*", content_range)) and len(sample) == 1


def download_hf_file_cpu(remote: dict, out: Path, receipt: dict) -> Path:
    """Full-file fallback only inside the already-guarded CPU PBS allocation."""
    cache_path = safe_cache_file(out, remote["source_repo_id"], remote["source_revision"], remote["source_repo_path"])
    known = set(receipt.get("downloaded_paths", []))
    if cache_path.is_file() and remote["source_repo_path"] in known:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + ".partial")
    lock_path = cache_path.with_name(cache_path.name + ".download.lock")
    try:
        lock_handle = lock_path.open("x", encoding="utf-8")
        lock_handle.write(f"pid={os.getpid()}\n")
        lock_handle.close()
    except FileExistsError:
        raise RuntimeError(f"Another CPU allocation is already downloading {remote['source_repo_path']}") from None
    request = Request(hf_repo_path(remote["source_repo_id"], remote["source_revision"], remote["source_repo_path"]),
                      headers={"User-Agent": "fastwam-vq-calibration-input-preparer/1.0"})
    print(f"CPU_FALLBACK_DOWNLOAD_STARTED repo_path={remote['source_repo_path']}", flush=True)
    total = 0
    try:
        with urlopen(request, timeout=90) as response, temporary.open("wb") as destination:
            while True:
                block = response.read(HTTP_BLOCK_SIZE)
                if not block:
                    break
                destination.write(block)
                total += len(block)
        expected_size = remote.get("dataset_size_bytes")
        if expected_size is not None and total != int(expected_size):
            raise RuntimeError(f"CPU source download size mismatch for {remote['source_repo_path']}")
        os.replace(temporary, cache_path)
    except RuntimeError:
        if temporary.exists():
            temporary.unlink()
        lock_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        lock_path.unlink(missing_ok=True)
        status = getattr(exc, "code", None)
        suffix = f" HTTP {status}" if status else f" ({type(exc).__name__})"
        raise RuntimeError(f"CPU PBS fallback download failed for {remote['source_repo_path']}" + suffix) from None
    lock_path.unlink(missing_ok=True)
    receipt.setdefault("downloaded_paths", []).append(remote["source_repo_path"])
    receipt["downloaded_paths"] = sorted(set(receipt["downloaded_paths"]))
    write_json_atomic(out / "source_receipt.json", receipt)
    print(f"CPU_FALLBACK_DOWNLOAD_COMPLETE repo_path={remote['source_repo_path']} bytes={total}", flush=True)
    return cache_path


@contextmanager
def open_hdf5(source: dict, out: Path, receipt: dict | None):
    import h5py

    if source.get("dataset_source") != "huggingface":
        with h5py.File(source["dataset_file"], "r") as h5:
            yield h5
        return
    if receipt is None or receipt.get("revision") != source.get("source_revision"):
        raise RuntimeError("Pinned Hugging Face source receipt is missing or does not match the frozen plan")
    repo_path = source["source_repo_path"]
    cached = safe_cache_file(out, source["source_repo_id"], source["source_revision"], repo_path)
    if cached.is_file() and repo_path in receipt.get("downloaded_paths", []):
        with h5py.File(cached, "r") as h5:
            yield h5
        return
    verified = set(receipt.get("range_verified_paths", []))
    unsupported = set(receipt.get("range_unsupported_paths", []))
    if repo_path not in verified and repo_path not in unsupported:
        if probe_hf_ranges(source):
            verified.add(repo_path)
            receipt["range_verified_paths"] = sorted(verified)
            receipt["range_support_confirmed"] = True
            write_json_atomic(out / "source_receipt.json", receipt)
        else:
            unsupported.add(repo_path)
            receipt["range_unsupported_paths"] = sorted(unsupported)
            receipt["range_support_confirmed"] = bool(verified)
            write_json_atomic(out / "source_receipt.json", receipt)
            print(f"RANGE_UNSUPPORTED using_cpu_download_fallback repo_path={repo_path}", flush=True)
            cached = download_hf_file_cpu(source, out, receipt)
            with h5py.File(cached, "r") as h5:
                yield h5
            return
    if repo_path in unsupported:
        cached = download_hf_file_cpu(source, out, receipt)
        with h5py.File(cached, "r") as h5:
            yield h5
        return

    import fsspec
    url = hf_repo_path(source["source_repo_id"], source["source_revision"], repo_path)
    try:
        with fsspec.open(url, mode="rb", block_size=HTTP_BLOCK_SIZE, cache_type="blockcache") as handle:
            with h5py.File(handle, "r") as h5:
                yield h5
    except Exception as exc:
        raise RuntimeError(f"Range-backed HDF5 read failed for pinned source {repo_path} ({type(exc).__name__})") from None


def inventory_task(suite_obj, suite_name: str, task_id: int, dataset_roots: list[Path],
                   out: Path, receipt: dict | None, hf_files: list[dict] | None,
                   repo_id: str, revision: str | None) -> tuple[dict | None, dict | None]:
    task = suite_obj.get_task(task_id)
    local_file, candidates = dataset_candidates(dataset_roots, suite_name, str(task.name))
    if local_file is not None:
        source = {"dataset_source": "local", "dataset_file": str(local_file), "source_repo_id": None,
                  "source_revision": None, "source_repo_path": None, "dataset_size_bytes": local_file.stat().st_size}
    else:
        remote_item = find_hf_dataset_file(hf_files or [], suite_name, str(task.name))
        if remote_item is None or revision is None:
            return None, {"suite": suite_name, "task_id": task_id, "task_name": str(task.name),
                          "expected_candidates": candidates,
                          "expected_hf_path": f"{suite_name}/<canonical-task-name>_demo.hdf5"}
        source = remote_source(repo_id, revision, remote_item)
    demos: dict[int, int] = {}
    with open_hdf5(source, out, receipt) as h5:
        data = h5.get("data")
        if data is None:
            raise RuntimeError(f"Original HDF5 lacks the official data group: {source['dataset_file']}")
        for name in data.keys():
            match = re.fullmatch(r"demo_(\d+)", str(name))
            if not match:
                continue
            states = data[name].get("states")
            if states is None or len(states.shape) != 2 or int(states.shape[0]) < 3:
                raise RuntimeError(f"Invalid flattened states in original HDF5: {source['dataset_file']}:/data/{name}/states")
            demos[int(match.group(1))] = int(states.shape[0])
    if len(demos) < 3:
        raise RuntimeError(f"Need three original demonstrations in {source['dataset_file']}; found {len(demos)}")
    return ({"suite": suite_name, "task_id": task_id, "task_name": str(task.name),
             "description": str(task.language), **source, "demos": demos}, None)


def source_file(module) -> Path:
    value = getattr(module, "__file__", None)
    if not value:
        raise RuntimeError(f"Imported module has no source path: {module.__name__}")
    return Path(value).resolve()


def prepare_environment(fastwam_root: Path, source_root: Path, compat: Path, config: Path):
    import numpy as np

    if "float_" not in np.__dict__:
        np.float_ = np.float64
    if np.float_ is not np.float64:
        raise RuntimeError("LIBERO compatibility requires numpy.float_ to alias float64")
    prefixes = [compat, source_root, fastwam_root / "FastWAM", fastwam_root / "FastWAM/src",
                fastwam_root / "FastWAM/experiments/libero"]
    os.environ["PYTHONPATH"] = os.pathsep.join(map(str, prefixes))
    os.environ["LIBERO_CONFIG_PATH"] = str(config)
    os.environ["ROOT"] = str(fastwam_root)
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    sys.path[:0] = list(map(str, prefixes))

    import libero
    import libero.libero as libero_core
    from libero.libero import benchmark
    from libero.libero import envs as envs_module
    from experiments.libero import libero_utils as ev

    assets_root = Path(libero_core.get_libero_path("assets")).resolve()
    if not assets_root.is_dir():
        raise FileNotFoundError(f"Configured original LIBERO assets directory is unavailable: {assets_root}")
    provenance = {"libero_outer": str(source_file(libero)), "libero_core": str(source_file(libero_core)),
                  "benchmark": str(source_file(benchmark)), "envs": str(source_file(envs_module)),
                  "libero_utils": str(source_file(ev)), "source_root": str(source_root),
                  "compat_prefix": str(compat), "assets_root": str(assets_root)}
    resolved_source, resolved_compat = source_root.resolve(), compat.resolve()
    for key in ("benchmark", "envs"):
        Path(provenance[key]).resolve().relative_to(resolved_source)
    for key in ("libero_outer", "libero_core"):
        path = Path(provenance[key]).resolve()
        if not path.is_relative_to(resolved_source) and not path.is_relative_to(resolved_compat):
            raise RuntimeError(f"LIBERO import escaped declared source/compat roots: {key}={path}")
    Path(provenance["libero_utils"]).resolve().relative_to((fastwam_root / "FastWAM/experiments/libero").resolve())
    return np, benchmark, ev, provenance


def render_one(row: dict, out: Path, runtime, suites: dict, provenance: dict,
               fastwam_root: Path, receipt: dict | None) -> None:
    from libero.libero.utils.utils import postprocess_model_xml

    np, ev = runtime
    suite = suites[row["suite"]]
    task_id = int(row["task_id"])
    task = suite.get_task(task_id)
    if str(task.name) != row["task_name"]:
        raise RuntimeError(f"Task identity changed for {row['suite']} task {task_id}")
    state_path = f"{row['dataset_file']}::/data/demo_{row['demo_id']}/states[{row['frame_index']}]"
    with open_hdf5(row, out, receipt) as h5:
        demo_group = h5[f"data/demo_{int(row['demo_id'])}"]
        dataset = demo_group["states"]
        if int(dataset.shape[0]) != int(row["demo_length"]):
            raise RuntimeError(f"Demo length changed since plan freeze: {state_path}")
        state = np.asarray(dataset[int(row["frame_index"])])
        model_xml_value = demo_group.attrs.get("model_file")
    if state.ndim != 1 or state.dtype.kind not in "fiu" or not np.isfinite(state).all():
        raise RuntimeError(f"Expected a finite flattened simulator state: {state_path}")
    if isinstance(model_xml_value, bytes):
        model_xml_value = model_xml_value.decode("utf-8")
    else:
        model_xml_value = str(model_xml_value) if model_xml_value is not None else ""
    if not model_xml_value.strip():
        raise RuntimeError(f"Selected original demonstration has no model_file XML attribute: {state_path}")
    model_xml = postprocess_model_xml(model_xml_value, {})
    model_xml, relocated_assets = relocate_libero_asset_paths(model_xml, Path(provenance["assets_root"]))
    for relative_asset, is_directory in relocated_assets:
        asset_path = Path(provenance["assets_root"]).joinpath(*PurePosixPath(relative_asset).parts)
        exists = asset_path.is_dir() if is_directory else asset_path.is_file()
        if not exists:
            raise FileNotFoundError(f"Original LIBERO asset referenced by demo XML is missing: {relative_asset}")

    environment_seed = int(row["environment_seed"])
    if environment_seed != 830_000_000 + int(row["case_id"]) * 17:
        raise RuntimeError(f"Frozen environment_seed changed for case {row['case_id']}")
    random.seed(environment_seed)
    np.random.seed(environment_seed % (2**32))
    env = ev.OffScreenRenderEnv(bddl_file_name=str(suite.get_task_bddl_file_path(task_id)),
                                camera_heights=int(ev.LIBERO_ENV_RESOLUTION),
                                camera_widths=int(ev.LIBERO_ENV_RESOLUTION))
    try:
        env.seed(environment_seed)
        env.reset()
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
        observation = env.set_init_state(state)
        arrays = {str(key): np.asarray(value) for key, value in observation.items()
                  if np.asarray(value).dtype.kind in "biufc"}
        missing = set(REQUIRED_OBSERVATIONS) - arrays.keys()
        if missing:
            raise RuntimeError(f"Rendered observation lacks required fields: {sorted(missing)}")
        resolution = int(ev.LIBERO_ENV_RESOLUTION)
        for key in REQUIRED_OBSERVATIONS[:2]:
            if arrays[key].shape != (resolution, resolution, 3):
                raise RuntimeError(f"Unexpected rendered {key} shape: {arrays[key].shape}")
        for key, value in arrays.items():
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise RuntimeError(f"Raw observation {key} is not finite numeric data")

        obs_path = out / row["observation_file"]
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_npz = obs_path.with_name(obs_path.name + ".tmp")
        with temporary_npz.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_npz, obs_path)
        metadata = {
            **{key: row[key] for key in ("case_id", "suite", "task_id", "task_name", "domain", "phase",
                                         "demo_id", "frame_index", "sampler_seeds", "observation_file",
                                         "metadata_file", "dimension", "variant", "variant_id", "original_task",
                                         "source_index", "environment_seed")},
            "description": str(task.language),
            "statepath": state_path, "settling_steps": 0, "evaluation_episodes": 0,
            "episode_count": 0,
            "input_identity": row_identity(row),
            "source_provenance": {**source_provenance(row), "state_dataset": "/data/demo_N/states",
                                  "demo_model_xml_attribute": "data/demo_N.attrs[model_file]",
                                  "demo_model_xml_restore": "postprocess_model_xml+reset_from_xml_string+sim.reset",
                                  "demo_model_xml_asset_relocation": "chiliocosm/assets/* -> configured original LIBERO assets root",
                                  "snapshot_restore": "demo model XML reset, then OffScreenRenderEnv.set_init_state",
                                  "state_loader": "original LIBERO demonstration HDF5"},
            "dataset_import_paths": {**provenance, "base_root": str(fastwam_root)},
            "observation_arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                   for key, value in arrays.items()},
        }
        write_json_atomic(out / row["metadata_file"], metadata)
    finally:
        env.close()


def validate_plan(plan: dict) -> None:
    if plan.get("protocol") != PROTOCOL or plan.get("selection_seed") != SELECTION_SEED:
        raise RuntimeError("Existing plan belongs to a different calibration protocol or selection seed")
    rows = plan.get("inputs")
    if not isinstance(rows, list) or len(rows) != 120:
        raise RuntimeError("Frozen plan must contain exactly 80 calibration and 40 test contexts")
    calibration = [row for row in rows if 2000 <= int(row.get("case_id", -1)) < 2080]
    tests = [row for row in rows if 3000 <= int(row.get("case_id", -1)) < 3040]
    cohorts = plan.get("calibration_cohorts", {})
    if ([row["case_id"] for row in calibration] != list(range(2000, 2080))
            or [row["case_id"] for row in tests] != list(range(3000, 3040))
            or len(cohorts.get("c8", [])) != 8 or len(cohorts.get("c80", [])) != 80
            or set(cohorts["c80"]) != {row["case_id"] for row in calibration}
            or not set(cohorts["c8"]) <= set(cohorts["c80"])):
        raise RuntimeError("Existing plan cohort membership is malformed")
    if any(row.get("domain") != "original" for row in rows):
        raise RuntimeError("The calibration-suite plan must contain original-domain inputs only")
    for suite in SUITES:
        suite_cal = [row for row in calibration if row["suite"] == suite]
        suite_test = [row for row in tests if row["suite"] == suite]
        if len(suite_cal) != 20 or len(suite_test) != 10:
            raise RuntimeError(f"Existing plan has unexpected counts for {suite}")
        if [sum(row.get("phase") == phase for row in suite_cal) for phase in PHASES] != [7, 7, 6]:
            raise RuntimeError(f"Existing plan has unexpected C80 phase allocation for {suite}")
        suite_c8 = [row for row in suite_cal if row["case_id"] in cohorts["c8"]]
        if len(suite_c8) != 2 or len({row["task_id"] for row in suite_c8}) != 2:
            raise RuntimeError(f"Existing plan C8 must contain two different tasks for {suite}")
    for row in rows:
        if row.get("dimension") != "original" or row.get("variant") != "original":
            raise RuntimeError("Plan input identity must remain in the original LIBERO domain")
        if int(row.get("environment_seed", -1)) != 830_000_000 + int(row["case_id"]) * 17:
            raise RuntimeError(f"Invalid frozen environment seed for case {row['case_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--out", type=Path, help="Data root; plan.json and inputs/ are written here")
    parser.add_argument("--control-test", action="store_true")
    args = parser.parse_args()
    if args.control_test:
        control_test()
        return 0
    if args.protocol is None or args.out is None:
        parser.error("--protocol and --out are required unless --control-test is used")

    allocation_guard()
    job_started = time.monotonic()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol") != PROTOCOL:
        raise RuntimeError(f"Expected {PROTOCOL}, got {protocol.get('protocol')!r}")
    data_protocol = protocol.get("data", {})
    seed = int(data_protocol.get("selection_seed", protocol.get("selection_seed", SELECTION_SEED)))
    if seed != SELECTION_SEED:
        raise RuntimeError(f"Protocol selection_seed must be {SELECTION_SEED}")
    # Distinct, recorded seed ranges for calibration and held-out queries.
    seed_base = int(protocol.get("sampler_seed_base", 910_000_000))
    test_seed_base = int(protocol.get("test_sampler_seed_base", seed_base + 1_000_000))
    calibration_start = int(data_protocol.get("calibration_case_id_start", 2000))
    test_start = int(data_protocol.get("test_case_id_start", 3000))
    if (calibration_start, test_start) != (2000, 3000):
        raise RuntimeError("Calibration/test case IDs are frozen at 2000 and 3000")
    pairs = [tuple(pair) for pair in data_protocol.get("calibration_task_phase_pairs", PHASE_PAIRS)]
    test_phases = list(data_protocol.get("test_phases", ["early"] * 3 + ["middle"] * 4 + ["late"] * 3))
    phase_positions = data_protocol.get("phase_positions", {"early": 0.2, "middle": 0.5, "late": 0.8})
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"

    receipt = None
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        validate_plan(plan)
    else:
        fastwam_root, source_root, compat, config, datasets = load_roots(protocol)
        np, benchmark, ev, provenance = prepare_environment(fastwam_root, source_root, compat, config)
        classes = benchmark.get_benchmark_dict()
        suites = {name: classes[name]() for name in SUITES}
        repo_id = str(data_protocol.get("source_hf_repo_id", DEFAULT_HF_REPO))
        missing_local = any(dataset_candidates(datasets, suite_name, str(suites[suite_name].get_task(task_id).name))[0] is None
                            for suite_name in SUITES for task_id in range(10))
        hf_files = None
        revision = None
        if missing_local:
            receipt = load_source_receipt(out, repo_id, create=True)
            revision = receipt["revision"]
            hf_files = hf_tree_files(repo_id, revision)
        tasks, missing = [], []
        for suite_name in SUITES:
            for task_id in range(10):
                task, absent = inventory_task(suites[suite_name], suite_name, task_id, datasets,
                                              out, receipt, hf_files, repo_id, revision)
                if absent:
                    missing.append(absent)
                else:
                    tasks.append(task)
        if missing:
            report = {"status": "missing_original_hdf5", "domain": "original", "source_repo_id": repo_id,
                      "source_revision": revision, "missing": missing}
            write_json_atomic(out / "missing_datasets.json", report)
            print(f"DATASET_INVENTORY_MISSING count={len(missing)} report={out / 'missing_datasets.json'}", flush=True)
            for item in missing:
                print("MISSING_HDF5 " + json.dumps(item, ensure_ascii=False), flush=True)
            raise RuntimeError(f"Original LIBERO datasets are incomplete; see {out / 'missing_datasets.json'}")
        inputs, cohorts = freeze_inputs(tasks, seed, seed_base, test_seed_base, pairs, test_phases,
                                        phase_positions, calibration_start, test_start)
        plan = {
            "protocol": PROTOCOL, "status": "preparing", "selection_seed": seed,
            "suite_order": list(SUITES), "source": "original LIBERO HDF5 demonstration simulator states",
            "source_hf_repo_id": repo_id,
            "source_hf_revision": revision,
            "source_receipt_file": "source_receipt.json" if revision else None,
            "calibration_cohorts": cohorts,
            "inputs": inputs,
            "input_count": len(inputs), "calibration_count": 80, "test_count": 40,
            "test_phase_counts_per_suite": {phase: test_phases.count(phase) for phase in PHASES},
            "phase_positions": phase_positions,
            "sampler_seed_rule": {"calibration_base": seed_base, "test_base": test_seed_base,
                                  "formula": "base + case_id * 2 + seed_index"},
            "dataset_roots_in_priority_order": [str(path) for path in datasets],
            "episode_count": 0, "predicted_actions_executed": 0,
            "claim_scope": "C8 is nested in C80; shared held-out evaluation is one disjoint original demo snapshot per task.",
        }
        validate_plan(plan)
        write_json_atomic(plan_path, plan)

    # A previous invocation may have frozen the plan and finished only some contexts.
    fastwam_root, source_root, compat, config, _datasets = load_roots(protocol)
    if any(row.get("dataset_source") == "huggingface" for row in plan["inputs"]):
        receipt = load_source_receipt(out, str(plan.get("source_hf_repo_id", DEFAULT_HF_REPO)), create=False)
        if receipt is None or receipt.get("revision") != plan.get("source_hf_revision"):
            raise RuntimeError("Frozen plan requires its matching source_receipt.json; refusing to resolve mutable HF refs")
    pending = []
    for row in plan["inputs"]:
        obs_path, metadata_path = out / row["observation_file"], out / row["metadata_file"]
        if obs_path.is_file() and metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                identity = row_identity(row)
                if metadata.get("case_id") == row["case_id"] and metadata.get("input_identity") == identity:
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        pending.append(row)

    if pending:
        checkpoint = plan.get("checkpoint", {})
        prior_estimate = checkpoint.get("seconds_per_context_estimate")
        deadline = min(float(protocol.get("soft_deadline_seconds", 6300)), 7140.0)
        reserve = 300.0
        pending_ids = {row["case_id"] for row in pending}
        all_case_ids = {row["case_id"] for row in plan["inputs"]}
        np, benchmark, ev, provenance = prepare_environment(fastwam_root, source_root, compat, config)
        suites = {name: benchmark.get_benchmark_dict()[name]() for name in SUITES}
        runtime = (np, ev)
        render_durations = []
        for index, row in enumerate(pending, 1):
            if row.get("dataset_source") != "huggingface" and not Path(row["dataset_file"]).is_file():
                raise FileNotFoundError(f"Frozen source HDF5 is unavailable: {row['dataset_file']}")
            elapsed = time.monotonic() - job_started
            estimate = (sum(render_durations) / len(render_durations) if render_durations
                        else float(prior_estimate or 0.0))
            if elapsed + max(reserve, 2.0 * estimate) >= deadline:
                plan["status"] = "preparing"
                complete_ids = sorted(all_case_ids - pending_ids)
                plan["checkpoint"] = {"completed_case_ids": complete_ids,
                                       "completed_context_count": len(complete_ids),
                                       "seconds_per_context_estimate": estimate or None}
                write_json_atomic(plan_path, plan)
                print(f"CHECKPOINT_EXIT completed={len(complete_ids)} remaining={len(pending) - index + 1} "
                      f"estimated_seconds_per_context={estimate:.1f} deadline_seconds={deadline:.0f}", flush=True)
                return 0
            context_started = time.monotonic()
            render_one(row, out, runtime, suites, provenance, fastwam_root, receipt)
            render_durations.append(time.monotonic() - context_started)
            render_durations = render_durations[-5:]
            estimate = sum(render_durations) / len(render_durations)
            completed = sorted((all_case_ids - pending_ids) | {item["case_id"] for item in pending[:index]})
            plan["status"] = "preparing"
            plan["checkpoint"] = {"completed_case_ids": completed,
                                   "completed_context_count": len(completed),
                                   "seconds_per_context_estimate": estimate}
            write_json_atomic(plan_path, plan)
            print(f"INPUT_RENDERED case_id={row['case_id']} suite={row['suite']} task={row['task_name']} "
                  f"demo={row['demo_id']} phase={row['phase']} pending={index}/{len(pending)}", flush=True)
            remaining = len(pending) - index
    plan["status"] = "complete"
    plan["checkpoint"] = {"completed_case_ids": [row["case_id"] for row in plan["inputs"]],
                           "completed_context_count": len(plan["inputs"]),
                           "seconds_per_context_estimate": plan.get("checkpoint", {}).get(
                               "seconds_per_context_estimate")}
    write_json_atomic(plan_path, plan)
    print(f"INPUTS_READY path={plan_path} calibration=80 test=40 c8=8 c80=80 episodes=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
