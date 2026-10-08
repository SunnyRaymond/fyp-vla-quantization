"""Execute the navigator's quality_prepare command with argv-safe Windows quoting."""
import os
import shlex
import subprocess
import sys
from pathlib import Path

SKILL = Path(r"C:\Users\Raymond\.codex\skills\idea-spark")
sys.path.insert(0, str(SKILL))
os.environ["IDEASPARK_CROSS_RUN_DEDUP"] = "off"
from scripts.quality_flow import late_flow

branch = Path(sys.argv[1]).resolve()
action = late_flow(branch, SKILL)
commands = action.get("run", [])
if len(commands) != 1:
    raise SystemExit(f"Expected one prepare command, got: {action}")
argv = shlex.split(commands[0])
if "quality_prepare" not in argv:
    raise SystemExit(f"Next action is not quality_prepare: {action}")
subprocess.run([sys.executable, *argv[1:]], check=True)
