"""PBS worker entry point for one independent Cosmos-adapted baseline bank."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

from baseline_quant import ARMS, calibrate_baseline


def _require_compute_allocation():
    if not os.environ.get("PBS_JOBID") or not os.environ.get("PBS_NODEFILE"):
        raise RuntimeError("Baseline tensor preparation requires an approved PBS allocation")
    hostname = socket.gethostname().lower()
    if "login" in hostname:
        raise RuntimeError("Baseline tensor preparation is forbidden on an ASPIRE2A login node")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--soft-deadline-seconds", type=float, default=0)
    args = parser.parse_args(argv)
    _require_compute_allocation()
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    import unittest
    import test_baseline_contract
    checks = unittest.defaultTestLoader.loadTestsFromModule(test_baseline_contract)
    if not unittest.TextTestRunner(verbosity=1).run(checks).wasSuccessful():
        raise RuntimeError('Allocated-GPU baseline math checks failed')
    from common import prepare_runtime
    runtime = prepare_runtime(protocol, Path(args.root), Path(args.out))
    try:
        summary = calibrate_baseline(runtime, protocol, args.arm, Path(args.root),
                                     Path(args.out), args.soft_deadline_seconds)
        if summary.get("pbs_job_id") != os.environ["PBS_JOBID"]:
            raise RuntimeError("Baseline summary is not bound to the current PBS job")
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if summary.get("status") == "complete" else 2
    finally:
        close = getattr(runtime, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    raise SystemExit(main())
