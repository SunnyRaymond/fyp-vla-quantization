"""Trace slow runtime initialization without instrumenting measured forwards."""
from contextlib import contextmanager
from datetime import datetime, timezone
import faulthandler
import os
from pathlib import Path
import time


@contextmanager
def startup_trace(out):
    path = Path(out) / "startup_traceback.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with path.open("a", buffering=1) as stream:
        stream.write(f"START utc={datetime.now(timezone.utc).isoformat()} "
                     f"pid={os.getpid()} job={os.environ.get('PBS_JOBID')}\n")
        faulthandler.dump_traceback_later(120, repeat=True, file=stream)
        try:
            yield
        finally:
            faulthandler.cancel_dump_traceback_later()
            stream.write(f"END elapsed_seconds={time.monotonic()-started:.3f}\n")
