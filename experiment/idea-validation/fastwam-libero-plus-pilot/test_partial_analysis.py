"""Small paired-statistics check without experiments or SSH."""
from analyze_completed import summarize
from aggregate import ARMS

records = []
for index in range(2):
    for arm in ARMS:
        records.append(dict(variant_id=str(index), index=index, suite='suite', dimension='dim',
            task_id=0, env_seed=index, state_id=0, initial_state_path='fixed', description='fixed',
            original_task='task', arm=arm, success=(arm == 'bf16' or index == 0 and arm == 'w4a8'),
            termination='success' if arm == 'bf16' or index == 0 and arm == 'w4a8' else 'control_step_timeout',
            steps=1, replans=1, episode_seconds=1.0,
            first_action_chunks=[[[0.0 if arm == 'bf16' else 1.0] * 7 for _ in range(32)]]))
result = summarize(records, 'check', 1)
assert result['arms']['bf16']['successes'] == 2 and result['arms']['w4a8']['successes'] == 1
assert result['paired']['w4a4_vs_bf16']['reference_only_success'] == 2
assert result['first_action']['w4a4_vs_bf16']['10']['motor_rmse']['median'] == 1.0
records[-1]['env_seed'] = 99
try:
    summarize(records, 'check', 1)
except ValueError:
    pass
else:
    raise AssertionError('Unpaired seeds accepted')
print('PARTIAL_ANALYSIS_CONTROL_OK')
