"""Run the original skill CLI with two verified-unavailable sources disabled.

Run-local adapter only; does not edit the installed skill. See retrieval limitations.
Semantic Scholar: repeated HTTP429 + two timeouts. OpenReview: challenge-required403.
All real retrieval, matching, output generation and phase routing use original code.
"""
import runpy
import sys
import subprocess

for unavailable in ("scripts.search_semanticscholar", "scripts.search_openreview"):
    sys.modules[unavailable] = None

entry = r"C:\Users\Raymond\.codex\skills\idea-spark\scripts\run.py"
# quality_collision starts a fresh run.py process. Carry this same policy
# into that child; the original connector and receipt code remain in use.
_original_subprocess_run = subprocess.run
def _available_connector_run(*args, **kwargs):
    call_args = list(args)
    if call_args and isinstance(call_args[0], (list, tuple)):
        command = list(call_args[0])
        # Availability is probed in another interpreter with --help. A local
        # sys.modules entry cannot disable that probe, so return the already
        # established unavailable status for these exact connector modules.
        if (len(command) > 2 and command[1] == '-m'
                and command[2] in ('scripts.search_semanticscholar', 'scripts.search_openreview')):
            return subprocess.CompletedProcess(command, 2, stdout='', stderr=
                'Run-local source skip after observed HTTP429/timeouts or challenge-required403; see RETRIEVAL_LIMITATIONS.zh.md')
        if (len(command) > 2
                and str(command[1]).replace('\\', '/').casefold() == entry.replace('\\', '/').casefold()
                and command[2] == 'phase3_collision'):
            command[1] = __file__
            call_args[0] = command
    return _original_subprocess_run(*call_args, **kwargs)
subprocess.run = _available_connector_run
sys.argv = [entry, *sys.argv[1:]]
runpy.run_path(entry, run_name="__main__")
