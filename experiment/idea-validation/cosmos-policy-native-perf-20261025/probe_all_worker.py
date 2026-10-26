"""Run all primitive diagnostics with one Torch initialization, without a policy."""
import argparse
import json
import os
from pathlib import Path

import diagnose_atom
import diagnose_nunchaku_pack
import kernel_probe
import native_diffusion
from prepare_sources import guard
from startup_trace import startup_trace


def _run_stages(root, torch, device):
    """The entry point verifies the real allocation before calling this helper."""
    out = root / 'results/preparation'
    out.mkdir(parents=True, exist_ok=True)
    report = {'format': 'cosmos_native_probe_driver_v1', 'status': 'running',
              'pbs_job_id': os.environ.get('PBS_JOBID'), 'stages': {},
              'policy_forward_calls': 0, 'training_updates': 0, 'closed_loop_episodes': 0}
    report_path = out / 'probe_all_driver.json'
    kernel_probe._write_probe_receipt(report_path, report)
    try:
        nunchaku = diagnose_nunchaku_pack.run(root)
        kernel_probe._write_probe_receipt(out / 'nunchaku_packing_diagnostic.json', nunchaku)
        report['stages']['nunchaku'] = nunchaku['status']
        if nunchaku['status'] != 'complete':
            raise RuntimeError('Nunchaku layout diagnostic failed; malformed operands must not reach GEMM')
        original = kernel_probe._extension('viditq_atom_native',
                                           root / 'native_extensions/viditq_w4a4')
        general = native_diffusion._extension('viditq_w4a4', root)
        for label, extension in (('original', original), ('general', general)):
            path = out / ('atom_diagnostic_' + label + '.json')
            diagnostic = diagnose_atom.run(root, path, torch_module=torch, device=device,
                                            allocation_guarded=True,
                                            extension=extension, extension_label=label)
            report['stages']['atom_' + label] = {
                'status': diagnostic['status'], 'scope': 'diagnostic collection; inspect individual cases',
                'path': str(path)}
            kernel_probe._write_probe_receipt(report_path, report)
        rc = kernel_probe.main(['--root', str(root), '--out',
            str(out / 'kernel_probe_native_all.json'), '--require-all'])
        report['status'] = 'complete' if rc == 0 else 'failed'
        report['all_native_probe_return_code'] = rc
        kernel_probe._write_probe_receipt(report_path, report)
        return rc
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        kernel_probe._write_probe_receipt(report_path, report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(argv)
    guard()
    root = args.root.resolve()
    with startup_trace(root / 'results/preparation/probe_all_startup'):
        import torch
    device = torch.device('cuda', torch.cuda.current_device())
    kernel_probe._hardware(torch, device)
    return _run_stages(root, torch, device)


if __name__ == '__main__':
    raise SystemExit(main())
