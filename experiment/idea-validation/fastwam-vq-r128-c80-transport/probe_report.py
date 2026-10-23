"""Read completed probes only; never substitute unfinished training arms."""
import json
import os
from pathlib import Path
import socket
import sys
from datetime import datetime, timezone

import aggregate as a


def build(protocol, records, pair_records):
    suites, _, _, fraction, latent_weight = a._check_protocol(protocol)
    arms = protocol['probe_arms']
    if set(records) != set(arms):
        raise ValueError('Exactly the eight completed probe arms are required')
    validated, aggregates, bindings = {}, {}, {}
    for arm in arms:
        validated[arm], aggregates[arm] = {}, {}
        for split in ('calibration', 'heldout'):
            cases, seeds, _ = a._expected(split)
            found, suite_map, seed_map = a._validate_records(
                a.PROTOCOL, records[arm][split], arm, split, cases, seeds, suites, latent_weight)
            binding = (suite_map, seed_map)
            if split in bindings and bindings[split] != binding:
                raise ValueError('Probe cohorts are not paired')
            bindings[split] = binding
            validated[arm][split] = found
            aggregates[arm][split] = a._aggregate(found, cases, seeds, suites)
    pairs = a._validate_pair_records(pair_records, validated, suites)
    comparisons = {}
    for split in ('calibration', 'heldout'):
        values = {arm: aggregates[arm][split] for arm in arms}
        specs = list(a.COMPARISON_PAIRS[3:]) + [
            ('Joint_A4_vs_Seq_A4', 'Seq_A4', 'Joint_A4'),
            ('Joint_noA4_vs_Seq_noA4', 'Seq_noA4', 'Joint_noA4'),
            ('Joint_to_Seq_A4_vs_Seq_A4', 'Seq_A4', 'Joint_to_Seq_A4'),
            ('Seq_to_Joint_A4_vs_Joint_A4', 'Joint_A4', 'Seq_to_Joint_A4')]
        comparisons[split] = {
            name: a._compare(name, base, target, values, suites) for name, base, target in specs}
    held = comparisons['heldout']
    gap = held['Joint_A4_vs_Seq_A4']['overall']['absolute_delta']['mixed_loss']
    residual = {route: held[f'{route}_noA4_vs_{anchor}_noA4']['overall']['absolute_delta']['mixed_loss']
                for route, anchor in (('Seq_to_Joint', 'Seq'), ('Joint_to_Seq', 'Joint'))}
    maximum = max(abs(v) for v in residual.values())
    geometry = dict(original_A4_gap=gap, noA4_mixed_delta=residual,
                    max_absolute_noA4_delta=maximum, allowed_residual=fraction * gap if gap > 0 else None,
                    residual_to_gap_fraction=maximum / gap if gap > 0 else None,
                    passed=gap > 0 and maximum <= fraction * gap)
    descriptive = {}
    for split in comparisons:
        descriptive[split] = {}
        for route, anchor in (('Seq_to_Joint', 'Seq'), ('Joint_to_Seq', 'Joint')):
            on = comparisons[split][f'{route}_A4_vs_{anchor}_A4']['overall']['absolute_delta']
            off = comparisons[split][f'{route}_noA4_vs_{anchor}_noA4']['overall']['absolute_delta']
            descriptive[split][route] = {m: on[m] - off[m] for m in a.METRICS}
    for splits in aggregates.values():
        for value in splits.values():
            del value['_by_case']
    return dict(protocol=a.PROTOCOL, complete_probes=True, training_attribution='pending',
                aggregation=a.AGGREGATION, arms=aggregates, comparisons=comparisons,
                geometry=geometry, pair_residuals=a._residual_summary(pairs),
                descriptive_A4_minus_noA4_route_deltas=descriptive,
                scope='fixed-input decoded-BF16 transport; no new forward; not a training counterfactual')


def self_check():
    from test_aggregate import synthetic_inputs, PROTOCOL
    records, pairs = synthetic_inputs()
    full = a.build_comparison(PROTOCOL, records, pairs, {})
    result = build(PROTOCOL, {k: records[k] for k in PROTOCOL['probe_arms']}, pairs)
    for arm in PROTOCOL['probe_arms']:
        for split in ('calibration', 'heldout'):
            assert result['arms'][arm][split]['overall'] == full[split][arm]['overall']
    for name, _, _ in a.COMPARISON_PAIRS[3:]:
        assert result['comparisons']['heldout'][name] == full['comparisons'][name]
    assert result['geometry']['passed'] == full['decision']['geometry_attribution']['pass']
    assert result['training_attribution'] == 'pending'
    assert len(json.dumps(result, ensure_ascii=False, separators=(',', ':')).encode('utf-8')) < 256 * 1024


def main():
    self_check()
    if sys.argv[1:] == ['--self-check']:
        print('Synthetic probe-only aggregation matches the existing full aggregator')
        return
    root = Path(sys.argv[1])
    nodefile = Path(os.environ['PBS_NODEFILE'])
    host = socket.gethostname().split('.')[0]
    assert os.environ.get('PBS_JOBID') and 'login' not in host
    assert host in [x.split('.')[0] for x in nodefile.read_text().split()]
    read = lambda p: json.loads(p.read_text())
    protocol = read(root / 'protocol.json')
    out = root / 'results'
    for stage in ('probe_0', 'probe_1', 'gate_0'):
        receipt = read(out / 'stages' / stage / 'summary.json')
        assert receipt['complete'] and receipt['protocol'] == a.PROTOCOL
    records = {}
    for arm in protocol['probe_arms']:
        records[arm] = {}
        for split in ('calibration', 'heldout'):
            cases, seeds, _ = a._expected(split)
            records[arm][split] = [read(out / 'endpoints' / arm / split / f'{c}_{s}.json')
                                    for c in cases for s in seeds]
    pairs = []
    for anchor in ('Seq', 'Joint'):
        for split in ('calibration', 'heldout'):
            cases, seeds, _ = a._expected(split)
            pairs.extend(read(out / 'pairs' / anchor / split / f'{c}_{s}.json') for c in cases for s in seeds)
    result = build(protocol, records, pairs)
    result['provenance'] = dict(aggregate_job=os.environ['PBS_JOBID'], host=host,
                               created_utc=datetime.now(timezone.utc).isoformat(),
                               new_teacher_queries=0, new_student_queries=0)
    target = root / 'probe_analysis' / 'probe_comparison.json'
    temporary = target.with_suffix('.tmp')
    serialized = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    assert len(serialized.encode('utf-8')) < 256 * 1024
    temporary.write_text(serialized, encoding='utf-8')
    temporary.replace(target)
    print(json.dumps({'complete': True, 'geometry': result['geometry'], 'path': str(target)}))


if __name__ == '__main__':
    main()
