"""Light control check; no SSH, model, or experiment runs."""
from pathlib import Path
from submit_array import load_range_plan, expected_config, range_window
from array_sources import range_child_source

plan = load_range_plan(Path(__file__).with_name('shard_ranges.json'))
ranges = plan['ranges']
assert len(ranges) == 512 and ranges[0]['start'] == 0 and ranges[-1]['stop'] == 1400
for first in (0, 90):
    config = expected_config('check', None, 10, plan, first, 90)
    assert (config['array_first'], config['array_last']) == (first, first + 89)
    rows = [range_child_source(i, ranges) for i in range(first, first + 90)]
    assert [r['pbs_array_index'] for r in rows] == list(range(first, first + 90))
    assert sum(r['expected_episode_count'] for r in rows) == 4 * sum(
        ranges[i]['stop'] - ranges[i]['start'] for i in range(first, first + 90))
assert expected_config('full', None, 10, plan)['array_last'] == 511
assert expected_config('static', 4, 10)['array_last'] == 349
for first, count in ((True, 1), (0, True), (512, 1), (500, 13), (0, 0)):
    try:
        range_window(plan, first, count)
    except ValueError:
        pass
    else:
        raise AssertionError((first, count))
print('WINDOW_CONTROL_OK')

# Exercise the source writer without SSH or reading experiment artifacts.
import contextlib
import io
import json
import sys
import tempfile
from unittest.mock import patch
import array_sources
import submit_array

with tempfile.TemporaryDirectory() as folder:
    handles, output = Path(folder) / 'handle.json', Path(folder) / 'sources.json'
    config = expected_config('check', None, 10, plan, 0, 1)
    config.update(state='confirmed', pbs_jobid='12345678[].pbs101')
    handles.write_text(json.dumps(config), encoding='utf-8')
    argv = ['array_sources.py', '--attempt', 'check', '--handles', str(handles),
            '--output', str(output), '--performance-jobid', '12345679.pbs101']
    for state, status, complete in [('F', 0, True), ('X', 0, True),
                                   ('X', 9, True), ('X', None, False),
                                   ('Q', 0, False), (None, 0, False)]:
        row = {'pbs_jobid': '12345678[0].pbs101', 'state': state, 'exit_status': status}
        with patch.object(sys, 'argv', argv), \
                patch.object(array_sources.remote, 'connect', return_value=(None, None)), \
                patch.object(array_sources, 'read_compact_qstat',
                             return_value=(0, {0: [row]}, 'B', [], [], [])), \
                contextlib.redirect_stdout(io.StringIO()):
            array_sources.main()
        source = json.loads(output.read_text(encoding='utf-8'))
        assert source['scheduler_complete'] is complete
        assert source['shards'][0]['state'] == state
        assert source['shards'][0]['exit_status'] == status
    (Path(folder) / 'RUN_STATE.json').write_text(
        json.dumps({'new_shard_submission_allowed': False}), encoding='utf-8')
    with patch.object(sys, 'argv', ['submit_array.py', '--shard-size', '4', '--attempt', 'pausecheck']), \
            patch.object(submit_array.remote, 'HERE', Path(folder)), \
            patch.object(submit_array.remote, 'connect', return_value=(None, None)), \
            patch.object(submit_array, 'qselect_ids', return_value=[]):
        try:
            submit_array.main()
        except RuntimeError as exc:
            assert 'paused by the user' in str(exc)
        else:
            raise AssertionError('Paused new-shard submission accepted')
print('ARRAY_TERMINAL_CONTROL_OK')
