#!/usr/bin/env python3
"""Download and stage the official PushT assets inside one PBS allocation.

This helper is deliberately standard-library-only and compute-node-only.  It
does not calculate hashes or trust a guessed archive prefix.  Before writing
the final asset root it parses the ZIP member table, requires one unique
checkpoint model root and one unique dataset root, rejects unsafe members, and
extracts only those selected roots.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable


CHECKPOINT_URL = "https://osf.io/download/xvzs4/"
DATASET_URL = "https://osf.io/download/k2d8w/"
CHECKPOINT_ARCHIVE = "official-checkpoints.zip"
DATASET_ARCHIVE = "official-pusht-noise.zip"
MODEL_REQUIRED = {
    "hydra.yaml",
    "checkpoints/model_latest.pth",
}
DATASET_REQUIRED = {
    "train/states.pth",
    "train/rel_actions.pth",
    "train/seq_lengths.pkl",
    "train/velocities.pth",
    "train/obses/episode_000.mp4",
    "val/states.pth",
    "val/rel_actions.pth",
    "val/seq_lengths.pkl",
    "val/velocities.pth",
    "val/obses/episode_000.mp4",
}


def _require_compute_node() -> None:
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("PBS_JOBID is required; refusing asset preparation outside a PBS allocation")
    host = os.uname().nodename.lower()
    if any(token in host for token in ("login", "head", "submit")):
        raise RuntimeError(f"refusing probable login host: {host}")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument(
        "--archive-source-root",
        type=Path,
        default=None,
        help="read-only directory containing the two already downloaded official ZIPs",
    )
    return parser.parse_args()


def _safe_member(name: str) -> PurePosixPath | None:
    raw = name.replace("\\", "/")
    if raw.startswith("/"):
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    raw = raw.rstrip("/")
    if not raw:
        return None
    raw_parts = raw.split("/")
    if any(part in ("", ".", "..") for part in raw_parts):
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    normalized = posixpath.normpath(raw)
    if normalized in ("", ".") or normalized.startswith("/"):
        return None
    parts = PurePosixPath(normalized).parts
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    return PurePosixPath(*parts)


def _safe_infos(zfile: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    result: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    for info in zfile.infolist():
        relative = _safe_member(info.filename)
        if relative is None or info.is_dir():
            continue
        mode = (info.external_attr >> 16) & 0o170000
        if mode == 0o120000:
            raise ValueError(f"symbolic links are not allowed in ZIP assets: {info.filename!r}")
        result.append((info, relative))
    if not result:
        raise ValueError("ZIP archive contains no regular files")
    return result


def _prefix_candidates(
    paths: Iterable[PurePosixPath], required_suffixes: set[str]
) -> list[tuple[str, tuple[str, ...]]]:
    suffixes_by_prefix: dict[tuple[str, ...], set[str]] = {}
    for path in paths:
        parts = path.parts
        for prefix_len in range(len(parts)):
            prefix = parts[:prefix_len]
            suffixes_by_prefix.setdefault(prefix, set()).add(
                "/".join(parts[prefix_len:])
            )
    candidates = {
        prefix
        for prefix, suffixes in suffixes_by_prefix.items()
        if required_suffixes.issubset(suffixes)
    }
    return [("/".join(prefix), prefix) for prefix in sorted(candidates)]


def _choose_unique_prefix(
    infos: list[tuple[zipfile.ZipInfo, PurePosixPath]],
    required_suffixes: set[str],
    label: str,
    allowed_tail: tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
    candidates = _prefix_candidates((path for _, path in infos), required_suffixes)
    tail_length = len(allowed_tail)
    candidates = [
        candidate
        for candidate in candidates
        if len(candidate[1]) >= tail_length and candidate[1][-tail_length:] == allowed_tail
    ]
    if len(candidates) != 1:
        names = [name or "<archive-root>" for name, _ in candidates]
        raise RuntimeError(
            f"expected one unique {label} archive root, found {len(candidates)}: {names}"
        )
    return candidates[0]


def _download(url: str, destination: Path) -> None:
    if destination.exists() or destination.with_name(destination.name + ".partial").exists():
        raise FileExistsError(f"refusing to overwrite existing archive or partial: {destination}")
    partial = destination.with_name(destination.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "dino-wm-pusht-transfer/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _extract_selected(
    archive: Path,
    output_root: Path,
    infos: list[tuple[zipfile.ZipInfo, PurePosixPath]],
    selected_prefix: tuple[str, ...],
    target_prefix: tuple[str, ...],
) -> int:
    count = 0
    with zipfile.ZipFile(archive) as zfile:
        for info, path in infos:
            if path.parts[: len(selected_prefix)] != selected_prefix:
                continue
            suffix = path.parts[len(selected_prefix) :]
            if not suffix:
                continue
            destination = output_root.joinpath(*target_prefix, *suffix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".partial")
            if partial.exists() or destination.exists():
                raise FileExistsError(f"refusing to overwrite extracted asset: {destination}")
            with zfile.open(info, "r") as source, partial.open("wb") as handle:
                shutil.copyfileobj(source, handle, length=8 * 1024 * 1024)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, destination)
            count += 1
    if count == 0:
        raise RuntimeError(f"selected {selected_prefix!r} has no extractable files")
    return count


def _archive_infos(path: Path) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"downloaded asset is not a readable ZIP archive: {path}")
    with zipfile.ZipFile(path) as zfile:
        return _safe_infos(zfile)


def main() -> int:
    _require_compute_node()
    args = _args()
    stage_root = args.stage_root.resolve()
    asset_root = args.asset_root.resolve()
    if stage_root.exists() or asset_root.exists():
        raise FileExistsError("job-specific PushT staging path already exists; refusing a second writer")
    stage_root.mkdir(parents=True)
    download_root = stage_root / "downloads"
    download_root.mkdir()
    partial_root = stage_root / "extracted.partial"
    final_root = asset_root
    partial_root.mkdir()

    if args.archive_source_root is not None:
        archive_source_root = args.archive_source_root.resolve()
        checkpoint_archive = archive_source_root / CHECKPOINT_ARCHIVE
        dataset_archive = archive_source_root / DATASET_ARCHIVE
        for archive in (checkpoint_archive, dataset_archive):
            if not archive.is_file():
                raise FileNotFoundError(
                    f"required read-only archive is missing; refusing a new download: {archive}"
                )
        print(f"reusing read-only checkpoint archive: {checkpoint_archive}", flush=True)
        print(f"reusing read-only PushT dataset archive: {dataset_archive}", flush=True)
    else:
        checkpoint_archive = download_root / CHECKPOINT_ARCHIVE
        dataset_archive = download_root / DATASET_ARCHIVE
        print(f"downloading official checkpoint archive to {checkpoint_archive}", flush=True)
        _download(CHECKPOINT_URL, checkpoint_archive)
        print(f"downloading official PushT dataset archive to {dataset_archive}", flush=True)
        _download(DATASET_URL, dataset_archive)

    checkpoint_infos = _archive_infos(checkpoint_archive)
    dataset_infos = _archive_infos(dataset_archive)
    checkpoint_name, checkpoint_prefix = _choose_unique_prefix(
        checkpoint_infos, MODEL_REQUIRED, "outputs/pusht", ("outputs", "pusht")
    )
    dataset_name, dataset_prefix = _choose_unique_prefix(
        dataset_infos, DATASET_REQUIRED, "pusht_noise", ("pusht_noise",)
    )
    print(f"selected checkpoint archive root: {checkpoint_name or '<archive-root>'}", flush=True)
    print(f"selected dataset archive root: {dataset_name or '<archive-root>'}", flush=True)

    checkpoint_count = _extract_selected(
        checkpoint_archive,
        partial_root,
        checkpoint_infos,
        checkpoint_prefix,
        ("checkpoints", "outputs", "pusht"),
    )
    dataset_count = _extract_selected(
        dataset_archive,
        partial_root,
        dataset_infos,
        dataset_prefix,
        ("data", "pusht_noise"),
    )
    if final_root.exists():
        raise FileExistsError(f"asset root appeared during preparation: {final_root}")
    os.replace(partial_root, final_root)
    manifest = {
        "schema": "dino-wm-shared-prefix-cache.pusht-assets-ready",
        "checkpoint_archive": CHECKPOINT_ARCHIVE,
        "dataset_archive": DATASET_ARCHIVE,
        "checkpoint_source": CHECKPOINT_URL,
        "dataset_source": DATASET_URL,
        "archive_source_root": str(args.archive_source_root.resolve()) if args.archive_source_root else None,
        "checkpoint_archive_root": checkpoint_name or "",
        "dataset_archive_root": dataset_name or "",
        "checkpoint_files_extracted": checkpoint_count,
        "dataset_files_extracted": dataset_count,
        "hashes": "not computed by policy",
    }
    (final_root / "PUSHT_ASSETS_READY.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"asset preparation failed closed: {exc}", file=sys.stderr, flush=True)
        raise
