"""Extract only the 40 frozen demo states/XMLs on a CPU allocation."""
import os
from pathlib import Path
import time

import common


def main():
    common.guard()
    if os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
        raise RuntimeError('Source HDF5 preparation requires CPU allocation')
    import numpy as np
    import render_reference as render
    p = common.validate(common.read(Path(os.environ['ARTIFACTS']) / 'protocol.json'))
    root = Path(p['remote_root'])
    data = root / 'data'
    data.mkdir(parents=True, exist_ok=True)
    original_data = Path(p['calibration']['original_plan']).parent
    receipt = common.read(original_data / 'source_receipt.json')
    rows = common.expected_rows(p)
    completed = []
    for case in p['rollout']['visited_ids']:
        row = rows[case]
        out = data / 'source_snapshots' / str(case)
        out.mkdir(parents=True, exist_ok=True)
        if not (out / 'source.json').exists():
            if time.time() - float(os.environ['FW_JOB_START_EPOCH']) > 6600:
                raise TimeoutError('CPU source preparation deadline; completed states retained')
            with render.open_hdf5(row, data, receipt) as h5:
                group = h5[f"data/demo_{row['demo_id']}"]
                state = np.asarray(group['states'][row['frame_index']])
                xml = group.attrs['model_file']
            if isinstance(xml, bytes):
                xml = xml.decode('utf-8')
            if state.ndim != 1 or not np.isfinite(state).all() or not str(xml).strip():
                raise RuntimeError('Invalid frozen demonstration state/XML')
            with (out / 'state.npy.tmp').open('wb') as f:
                np.save(f, state, allow_pickle=False)
            (out / 'state.npy.tmp').replace(out / 'state.npy')
            common.save(out / 'source.json', {'plan_row': row, 'model_xml': str(xml),
                        'pbs_jobid': os.environ['PBS_JOBID'], 'source_frame_index': row['frame_index']})
        if common.read(out / 'source.json')['plan_row'] != row or not (out / 'state.npy').is_file():
            raise RuntimeError('Existing source snapshot binding differs')
        completed.append(case)
        common.save(root / 'results/sources/progress.json', {
            'protocol': p['protocol'], 'pbs_jobid': os.environ['PBS_JOBID'],
            'phase': 'extracting', 'complete': False, 'completed': len(completed), 'last_case': case})
        print(f'SOURCE_SNAPSHOT_COMPLETE case={case} count={len(completed)}', flush=True)
    common.save(data / 'source_receipt.json', receipt)
    common.save(root / 'results/sources/summary.json', {
        'protocol': p['protocol'], 'pbs_jobid': os.environ['PBS_JOBID'], 'complete': True,
        'source_contexts': 40, 'case_ids': completed, 'new_model_forwards': 0})


if __name__ == '__main__':
    main()
