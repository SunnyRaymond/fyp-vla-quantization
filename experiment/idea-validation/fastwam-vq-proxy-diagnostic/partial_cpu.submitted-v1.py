"""Localize frozen-bank changes using the saved case0 capture; PBS CPU only."""
import json
import os
from pathlib import Path
import time


def statistics(torch, vectors):
    energy = float(vectors.square().mean(1).sum())
    net_energy = float(vectors.sum(0).square().mean())
    return {'net_rms': net_energy ** 0.5, 'sum_component_energy': energy,
            'net_to_sum_energy_ratio': net_energy / energy if energy else None}


def run():
    import run_experiment as api
    api.guard()
    import torch
    import codebook
    import smooth_vq
    import run_diagnostic as driver
    torch.set_num_threads(8)
    toy = statistics(torch, torch.tensor([[1., -1.], [-1., 1.]], dtype=torch.float64))
    assert toy['net_rms'] == 0 and toy['sum_component_energy'] == 2
    assert statistics(torch, torch.zeros(1, 4))['net_to_sum_energy_ratio'] is None
    assert statistics(torch, torch.ones(2, 1))['net_to_sum_energy_ratio'] == 2
    x, a, b, bias = torch.tensor([[2., 3.]]), torch.tensor([[1., 4.]]), torch.tensor([[3., 2.]]), torch.tensor([5.])
    assert torch.equal(torch.nn.functional.linear(x, a - b),
                       torch.nn.functional.linear(x, a, bias) - torch.nn.functional.linear(x, b, bias))
    print('PARTIAL_CPU_SELF_CHECKS_PASSED', flush=True)
    root = Path(os.environ['DIAG_ROOT'])
    out = root / 'results' / ('partial_cpu-' + os.environ['PBS_JOBID'])
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    preflight = root / 'results/preflight-25754358.pbs101'
    summary = json.loads((preflight / 'summary.json').read_text())
    receipt = root / 'artifacts/25754358.pbs101'
    if (summary['job_id'] != '25754358.pbs101' or summary['status'] != 'complete'
            or summary['case_ids'] != [0] or summary['target_linears'] != 614
            or summary['probe_counts'] != {'old': 4, 'new': 16, 'total': 20}
            or summary['row_sets'] != {'old': 2, 'new': 2}
            or not summary['original_proxy_reproduction_passed']
            or not (receipt / 'PIPELINE_COMPLETE').is_file()
            or (receipt / 'exit_code.txt').read_text().strip() != '0'):
        raise ValueError('Verified preflight reference is required')
    case = json.loads((preflight / 'case_0.json').read_text())
    capture = torch.load(preflight / 'sensitivity/case_0.pt', map_location='cpu', weights_only=True, mmap=True)
    if len(capture['layers']) != 614 or capture['probe_vectors'].shape != (20, 10, 6):
        raise ValueError('Saved capture coverage differs')
    protocol = json.loads((root / 'protocol.json').read_text())
    source = Path(protocol['source_root']) / 'results/full'
    baseline = 'row_output_vq_a4'
    arms = [a for a in protocol['frozen_arms'] if a != baseline]
    seed_labels = [protocol['old_probe_seed'], *protocol['new_probe_seeds']]
    labels = {'old_probes_old_rows': ('old', slice(0, 4)),
              'new_probes_old_rows': ('old', slice(4, 20)),
              'old_probes_new_rows': ('new', slice(0, 4)),
              'new_probes_new_rows': ('new', slice(4, 20))}
    modules = {a: {row: [] for row in ('old', 'new')} for a in arms}
    call_energy = {a: {row: torch.zeros(20, dtype=torch.float64) for row in ('old', 'new')} for a in arms}
    names = []
    started = time.perf_counter()
    with torch.no_grad():
        for index, (name, calls) in enumerate(capture['layers'].items()):
            names.append(name)
            ref = torch.load(source / 'banks' / baseline / f'{index:04d}.pt', map_location='cpu', weights_only=True)
            if ref['module'] != name:
                raise ValueError('Canonical bank identity differs')
            reference_weight = codebook.decode_vq(ref['encoded']).float()
            prepared = [(c, api.qinput(torch, smooth_vq, c['inputs'], ref['transform']).float()) for c in calls]
            record = {'module_index': index, 'module': name, 'arms': {}}
            for arm in arms:
                bank = torch.load(source / 'banks' / arm / f'{index:04d}.pt', map_location='cpu', weights_only=True)
                if bank['module'] != name:
                    raise ValueError('Compared bank identity differs')
                for key, value in ref['transform'].items():
                    equal = torch.equal(value, bank['transform'][key]) if torch.is_tensor(value) else value == bank['transform'][key]
                    if not equal:
                        raise ValueError('Bank transforms differ; a common-coordinate comparison is required')
                difference = codebook.decode_vq(bank['encoded']).float() - reference_weight
                total = {row: torch.zeros(20, dtype=torch.float64) for row in ('old', 'new')}
                for c, x in prepared:
                    # Shared bias and teacher output cancel between the two frozen banks.
                    response = driver.project_error(torch, c, torch.nn.functional.linear(x, difference))
                    for row in ('old', 'new'):
                        v = response[row].double()
                        total[row] += v
                        call_energy[arm][row] += v.square()
                record['arms'][arm] = {row: v.tolist() for row, v in total.items()}
                for row in ('old', 'new'):
                    modules[arm][row].append(total[row])
                del bank, difference
            api.append(out / 'module_changes.jsonl', record)
            del prepared, ref, reference_weight
            if index % 100 == 0 or index == 613:
                api.save(out / 'progress.json', {'job_id': os.environ['PBS_JOBID'], 'status': 'running', 'modules_completed': index + 1})
                print('MODULES_COMPLETED', index + 1, flush=True)
    result = {}
    for arm in arms:
        result[arm] = {}
        for label, (row, sl) in labels.items():
            vectors = torch.stack(modules[arm][row])[:, sl]
            item = statistics(torch, vectors)
            item['sum_call_energy'] = float(call_energy[arm][row][sl].mean())
            item['module_to_call_energy_ratio'] = (item['sum_component_energy'] / item['sum_call_energy']
                                                   if item['sum_call_energy'] else None)
            all_net = torch.stack(modules[arm][row]).sum(0)
            expected = torch.tensor([v for seed in seed_labels
                        for v in case['arms'][arm]['row_sets'][row][str(seed)]['predicted']], dtype=torch.float64)
            expected -= torch.tensor([v for seed in seed_labels
                        for v in case['arms'][baseline]['row_sets'][row][str(seed)]['predicted']], dtype=torch.float64)
            gap = float((all_net[sl] - expected[sl]).square().mean().sqrt())
            allowed = max(1e-4, 0.01 * float(expected[sl].square().mean().sqrt()))
            item['cpu_gpu_reference_check'] = {'rmse': gap, 'allowed_rmse': allowed, 'passed': gap <= allowed}
            energy = vectors.square().mean(1)
            top = torch.argsort(energy, descending=True)[:10].tolist()
            item['top_modules_by_change_energy'] = [{'module': names[i], 'energy': float(energy[i])} for i in top]
            result[arm][label] = item
    output = {'status': 'complete', 'phase': 'partial_cpu', 'job_id': os.environ['PBS_JOBID'],
              'reference_job': '25754358.pbs101', 'case_ids': [0], 'target_linears': 614,
              'baseline_arm': baseline, 'compared_arms': arms, 'aggregate': result,
              'wall_seconds': time.perf_counter() - started, 'refit': False, 'gpu_requested': False,
              'limits': ['Frozen-bank difference localization, not absolute local output error or a new fitted method.',
                         'One case; no independent scene replication or closed-loop claim.',
                         'Net/component ratios can exceed one; cancellation can be physically valid.',
                         'New-probe RMS uses all 16 directions pooled, unlike the preflight mean of four batch RMS values.',
                         'CPU/GPU comparison has its own declared tolerance; original reproduction tolerance is unchanged.']}
    api.save(out / 'summary.json', output)
    if not all(v['cpu_gpu_reference_check']['passed'] for groups in result.values() for v in groups.values()):
        raise ValueError('CPU evaluator disagrees with the saved GPU bank difference; keep diagnostic artifacts')
    api.save(Path(os.environ['ARTIFACTS']) / 'summary.json', output)
    (Path(os.environ['ARTIFACTS']) / 'PIPELINE_COMPLETE').touch()
    print('PARTIAL_CPU_COMPLETE', flush=True)


if __name__ == '__main__':
    run()
