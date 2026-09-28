"""Generate the frozen 2D Shapes-compatible dataset inside a PBS allocation."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
try:
    from allocation_guard import ensure_allocation
except ImportError:
    GUARD_DIR = ROOT.parent / "block-local-action-dynamics"
    sys.path.insert(0, str(GUARD_DIR))
    from allocation_guard import ensure_allocation

COMMIT = "e944b24bcaa42d9ee847f30163437a50f0237aa0"
REPOSITORY = "https://github.com/tkipf/c-swm"
FILES = (
    "envs/block_pushing.py",
    "modules.py",
    "utils.py",
    "data_gen/env.py",
    "LICENSE",
)
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
SET1_RGB = (
    (228, 26, 28), (55, 126, 184), (77, 175, 74),
    (152, 78, 163), (255, 127, 0), (255, 255, 51),
    (166, 86, 40), (247, 129, 191), (153, 153, 153),
)
EVENT_NAMES = {0: "free", 1: "other-object-blocked", 2: "boundary-blocked"}


class BlockPushingCompat:
    """Minimal standalone form of the pinned C-SWM Shapes environment."""

    def __init__(self, rng, objects=None):
        self.rng = rng
        if objects is None:
            cells = rng.choice(25, 5, replace=False)
            self.objects = [(int(p) // 5, int(p) % 5) for p in cells]
        else:
            self.objects = [tuple(map(int, p)) for p in objects]

    def classify(self, action):
        obj = int(action) // 4
        direction = int(action) % 4
        r, c = self.objects[obj]
        dr, dc = DIRECTIONS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= 5 or nc < 0 or nc >= 5:
            return 2
        if any(i != obj and (nr, nc) == pos for i, pos in enumerate(self.objects)):
            return 1
        return 0

    def step(self, action):
        obj = int(action) // 4
        direction = int(action) % 4
        if not 0 <= obj < 5:
            raise ValueError(f"Action out of range: {action}")
        category = self.classify(action)
        if category == 0:
            r, c = self.objects[obj]
            dr, dc = DIRECTIONS[direction]
            self.objects[obj] = (r + dr, c + dc)
        return category

    def render(self):
        """Return RGB uint8 CHW image, matching source geometry and ordering."""
        import numpy as np

        image = np.zeros((50, 50, 3), dtype=np.uint8)
        for obj_id, (r, c) in enumerate(self.objects):
            r0, c0 = r * 10, c * 10
            color = np.asarray(SET1_RGB[obj_id], dtype=np.uint8)
            kind = obj_id % 3
            if kind == 0:
                rr, cc = np.ogrid[max(0, r0):min(50, r0 + 11), max(0, c0):min(50, c0 + 11)]
                mask = (rr - (r0 + 5)) ** 2 + (cc - (c0 + 5)) ** 2 <= 25
                patch = image[max(0, r0):min(50, r0 + 11), max(0, c0):min(50, c0 + 11)]
                patch[mask] = color
            elif kind == 1:
                for row in range(r0, min(50, r0 + 11)):
                    rel = row - r0
                    left = c0 + 5 - rel / 2
                    right = c0 + 5 + rel / 2
                    lo = max(0, int(-(-left // 1)))
                    hi = min(49, int(right // 1))
                    if lo <= hi:
                        image[row, lo:hi + 1] = color
            else:
                image[max(0, r0):min(50, r0 + 11), max(0, c0):min(50, c0 + 11)] = color
        return image.transpose(2, 0, 1)


def official_source(source_root):
    source_root.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for relpath in FILES:
        destination = source_root / relpath
        if not destination.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/tkipf/c-swm/{COMMIT}/{relpath}"
            temporary = destination.with_suffix(destination.suffix + ".part")
            try:
                with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as out:
                    out.write(response.read())
            except Exception:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Could not retrieve pinned C-SWM source from this compute allocation: {url}. "
                    "Do not fall back to a login-node download; investigate an approved compute-node route."
                )
            temporary.replace(destination)
        downloaded.append({"path": relpath, "bytes": destination.stat().st_size})
    return downloaded


def dependency_probe():
    result = {}
    for module, distribution in (
        ("numpy", "numpy"), ("PIL", "Pillow"), ("torch", "torch"),
        ("matplotlib", "matplotlib"), ("skimage", "scikit-image"),
        ("h5py", "h5py"),
    ):
        present = importlib.util.find_spec(module) is not None
        try:
            version = importlib.metadata.version(distribution) if present else None
        except importlib.metadata.PackageNotFoundError:
            version = "present-version-unavailable"
        result[module] = {"available": present, "version": version}
    for required in ("numpy", "PIL", "torch"):
        if not result[required]["available"]:
            raise RuntimeError(f"Required dependency {required} is absent from the configured environment; no installation was attempted.")
    return result


def mechanism_checks():
    import numpy as np

    checks = []

    def record(name, fn):
        try:
            detail = fn()
            checks.append({"name": name, "status": "pass", "detail": detail or "ok"})
        except Exception as exc:
            checks.append({"name": name, "status": "fail", "detail": f"{type(exc).__name__}: {exc}"})

    def mapping():
        assert 17 // 4 == 4 and 17 % 4 == 1
        assert DIRECTIONS[1] == (0, 1)
        return "action 17 selects object 4, direction +column"

    def free_move():
        env = BlockPushingCompat(np.random.default_rng(1), [(2, 2), (1, 0), (0, 1), (0, 2), (0, 3)])
        assert env.step(5) == 0
        assert env.objects[1] == (1, 1)
        return "selected object moved one free cell"

    def occupied_block():
        env = BlockPushingCompat(np.random.default_rng(1), [(2, 2), (0, 0), (0, 1), (0, 2), (0, 3)])
        before = list(env.objects)
        assert env.step(5) == 1
        assert env.objects == before
        return "occupied destination leaves every object unchanged"

    def boundary_block():
        env = BlockPushingCompat(np.random.default_rng(1), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
        before = list(env.objects)
        assert env.step(0) == 2
        assert env.objects == before
        return "out-of-bounds destination leaves every object unchanged"

    def event_classes():
        env = BlockPushingCompat(np.random.default_rng(1), [(2, 2), (2, 3), (0, 0), (0, 1), (4, 4)])
        assert env.classify(1) == 1
        assert env.classify(8) == 2
        assert env.classify(6) == 0
        return "free/object-blocked/boundary-blocked labels match transitions"

    def render_contract():
        env = BlockPushingCompat(np.random.default_rng(1), [(0, 0), (0, 2), (2, 2), (3, 3), (4, 4)])
        image = env.render()
        assert image.shape == (3, 50, 50) and image.dtype == np.uint8
        assert all(tuple(image[:, r * 10 + 5, c * 10 + 5]) == SET1_RGB[i]
                   for i, (r, c) in enumerate(env.objects))
        return "RGB CHW uint8 50x50 and all five center colors are visible"

    for name, fn in (
        ("action_mapping", mapping), ("free_move", free_move),
        ("other_object_collision", occupied_block), ("grid_boundary", boundary_block),
        ("event_classification", event_classes), ("renderer_shape_and_colors", render_contract),
    ):
        record(name, fn)
    return checks


def generate_split(name, episodes, split_seed, output):
    import numpy as np

    steps = 40
    obs = np.lib.format.open_memmap(
        output / f"{name}_obs.npy", mode="w+", dtype=np.uint8,
        shape=(episodes, steps + 1, 3, 50, 50),
    )
    actions = np.lib.format.open_memmap(
        output / f"{name}_action.npy", mode="w+", dtype=np.int64,
        shape=(episodes, steps),
    )
    positions = np.lib.format.open_memmap(
        output / f"{name}_position.npy", mode="w+", dtype=np.uint8,
        shape=(episodes, steps + 1, 5, 2),
    )
    events = np.lib.format.open_memmap(
        output / f"{name}_event.npy", mode="w+", dtype=np.uint8,
        shape=(episodes, steps),
    )
    counts = [0, 0, 0]
    for episode in range(episodes):
        env_seed, action_seed = np.random.SeedSequence([split_seed, episode]).spawn(2)
        env_rng = np.random.default_rng(env_seed)
        action_rng = np.random.default_rng(action_seed)
        env = BlockPushingCompat(env_rng)
        for t in range(steps + 1):
            obs[episode, t] = env.render()
            positions[episode, t] = env.objects
            if t == steps:
                break
            action = int(action_rng.integers(0, 20))
            category = env.step(action)
            actions[episode, t] = action
            events[episode, t] = category
            counts[category] += 1
    for array in (obs, actions, positions, events):
        array.flush()
    return {
        "episodes": episodes, "steps_per_episode": steps,
        "split_seed": split_seed,
        "event_counts": {EVENT_NAMES[i]: counts[i] for i in range(3)},
        "files": {
            f"{name}_obs.npy": {"shape": [episodes, steps + 1, 3, 50, 50], "dtype": "uint8", "bytes": (output / f"{name}_obs.npy").stat().st_size},
            f"{name}_action.npy": {"shape": [episodes, steps], "dtype": "int64", "bytes": (output / f"{name}_action.npy").stat().st_size},
            f"{name}_position.npy": {"shape": [episodes, steps + 1, 5, 2], "dtype": "uint8", "bytes": (output / f"{name}_position.npy").stat().st_size},
            f"{name}_event.npy": {"shape": [episodes, steps], "dtype": "uint8", "bytes": (output / f"{name}_event.npy").stat().st_size},
        },
    }


def save_preview(data_dir, run_output):
    import numpy as np
    from PIL import Image, ImageDraw

    events = np.load(data_dir / "train_event.npy", mmap_mode="r")
    obs = np.load(data_dir / "train_obs.npy", mmap_mode="r")
    actions = np.load(data_dir / "train_action.npy", mmap_mode="r")
    selected = [(0, 0, "initial")]
    for category, title in EVENT_NAMES.items():
        found = np.argwhere(events == category)
        if found.size:
            ep, t = map(int, found[0])
            selected.append((ep, t, title))
    tile = 170
    canvas = Image.new("RGB", (2 * tile, 2 * tile), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for index, (ep, t, title) in enumerate(selected):
        frame = np.transpose(obs[ep, t], (1, 2, 0))
        image = Image.fromarray(frame, mode="RGB").resize((150, 150), Image.Resampling.NEAREST)
        x, y = (index % 2) * tile + 10, (index // 2) * tile + 6
        canvas.paste(image, (x, y))
        if title == "initial":
            label = title
        else:
            action = int(actions[ep, t])
            label = f"{title}: obj={action // 4}, dir={action % 4}"
        draw.text((x, y + 151), label, fill=(0, 0, 0))
    path = run_output / "sample_observations.png"
    canvas.save(path, format="PNG", optimize=True)
    return {"path": path.name, "bytes": path.stat().st_size, "panels": len(selected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Prepared dataset directory on shared scratch")
    parser.add_argument("--run-output", type=Path, required=True, help="This PBS job's small summaries directory")
    args = parser.parse_args()

    identity = ensure_allocation(require_gpu=False)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["source"]["commit"] != COMMIT:
        raise ValueError("FREEZE.json source commit does not match prepare_data.py")
    dependencies = dependency_probe()
    output = args.output.resolve()
    run_output = args.run_output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_output.mkdir(parents=True, exist_ok=True)
    (run_output / "FREEZE.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    checks = mechanism_checks()
    if any(item["status"] != "pass" for item in checks):
        (run_output / "mechanism_tests.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError("A block-pushing mechanism self-check failed; see mechanism_tests.json")

    source_files = official_source(output / "official_source" / COMMIT)
    checks.append({"name": "pinned_source_snapshot", "status": "pass", "detail": f"commit={COMMIT}; files=" + ",".join(row["path"] for row in source_files)})

    split_config = config["data"]["episodes"]
    split_seeds = config["data"]["split_seed"]
    manifest_path = output / "manifest.json"
    existing = [path for path in output.iterdir() if path.name not in {"official_source"}]
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("protocol_id") != config["protocol_id"] or old.get("frozen_config") != config or old.get("complete") is not True:
            raise RuntimeError("Prepared directory contains a partial or different protocol; use an empty output path.")
        splits = old["splits"]
    elif existing:
        raise RuntimeError("Prepared directory is nonempty without a complete manifest; use an empty output path.")
    else:
        splits = {}
        for name in ("train", "dev", "test"):
            print(f"Generating {name}: {split_config[name]} episodes x 40 steps", flush=True)
            splits[name] = generate_split(name, split_config[name], split_seeds[name], output)
        manifest = {
            "protocol_id": config["protocol_id"], "source_commit": COMMIT,
            "frozen_config": config,
            "complete": True, "episodes_are_split_unit": True,
            "model_training_arrays": ["*_obs.npy", "*_action.npy"],
            "diagnostic_only_arrays": ["*_position.npy", "*_event.npy"],
            "splits": splits,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    sample = save_preview(output, run_output)
    (run_output / "mechanism_tests.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    summary = {
        "protocol_id": config["protocol_id"], "source_commit": COMMIT,
        "execution_identity": identity, "dependencies": dependencies,
        "compatibility_renderer": config["environment"]["compatibility_changes"],
        "official_source_files": source_files,
        "prepared_directory": str(output), "splits": splits,
        "sample_image": sample,
        "data_usage_boundary": "obs/action train model; position/event diagnostics only",
    }
    (run_output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    done = {
        "status": "complete", "protocol_id": config["protocol_id"],
        "source_commit": COMMIT, "splits": {k: v["episodes"] for k, v in splits.items()},
        "sample_bytes": sample["bytes"],
    }
    (run_output / "DONE.json").write_text(json.dumps(done, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(done), flush=True)


if __name__ == "__main__":
    main()
