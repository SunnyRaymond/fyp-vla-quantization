"""CPU-allocation reporting only; no model imports or new inference."""
import json
import os
from pathlib import Path
import socket
import statistics


def read(path):
    return json.loads(path.read_text())


def write(path, value, compact=False):
    path.write_text(json.dumps(value, indent=None if compact else 2, allow_nan=False,
                               separators=(',', ':') if compact else None) + '\n')


def stats(values):
    return {'count': len(values), 'minimum': min(values) if values else None,
            'median': statistics.median(values) if values else None,
            'maximum': max(values) if values else None}


def main():
    assert os.environ.get('PBS_JOBID') and 'login' not in socket.gethostname()
    assert stats([1, 3]) == {'count': 2, 'minimum': 1, 'median': 2.0, 'maximum': 3}
    root = Path(__file__).resolve().parent
    out = root / 'results'
    p, result, summary = [read(path) for path in
                          (root/'protocol.json', out/'comparison.json', out/'summary.json')]
    assert summary['complete'] and summary['records_completed'] == 600
    assert summary['new_model_queries'] == 566 and summary['training_updates'] == 0
    assert result['coverage']['complete_600_record_panel']
    assert result['sensitivity']['coverage']['complete_160_row_panel']
    inputs = read(out/'input_perturbations.json')
    for name, value in (('comparison', result), ('input_perturbations', inputs)):
        write(out/(name+'.compact.json'), value, compact=True)
        assert (out/(name+'.compact.json')).stat().st_size <= 262144
    base = result['endpoint']['base']['overall_case_weighted']
    arms = ('clip_only', 'joint')
    endpoint = {a: base[a+'_endpoint_case_weighted'] for a in arms}
    case_rows = []
    wrong = {a: [] for a in arms}
    biases = {a: {field: [[] for _ in range(6)]
                  for field in ('normalized_action', 'executed_command')} for a in arms}
    for case in p['case_ids']:
        metrics = read(out/'case_metrics'/('%s_base.json' % case))
        teacher = read(out/'records/bf16'/('%s_base.json' % case))
        row = {'case_id': case, 'suite': teacher['suite']}
        for arm in arms:
            measured = metrics[arm+'_endpoint']
            row[arm] = {'motor_rmse': measured['motor']['normalized_action']['overall_rmse'],
                        'gripper_mismatches': measured['gripper']['command_sign_disagreement']['count'],
                        'gripper_normalized_rmse': measured['gripper']['normalized_rmse']}
            for field in biases[arm]:
                for d in range(6):
                    biases[arm][field][d].append(abs(measured['motor'][field]['per_dimension'][d]['signed_bias']))
            student = read(out/'records'/arm/('%s_base.json' % case))
            for step in range(10):
                target, actual = teacher['commands'][step][6], student['commands'][step][6]
                if target != actual:
                    wrong[arm].append({'case_id': case, 'step': step, 'teacher_command': target,
                                       'student_command': actual,
                                       'teacher_threshold_margin': abs(teacher['continuous_commands'][step][6])})
        case_rows.append(row)
    diagnostic = {
        'complete': True, 'pbs_jobid': os.environ['PBS_JOBID'], 'source_gpu_job': summary['pbs_jobid'],
        'new_model_queries': 0, 'training_updates': 0, 'new_rollouts': 0,
        'case_count': len(case_rows), 'case_rows': case_rows,
        'joint_vs_clip_base_motor': {
            key: sum(predicate(r['joint']['motor_rmse'], r['clip_only']['motor_rmse']) for r in case_rows)
            for key, predicate in (('improved', lambda j,c: j<c), ('worsened', lambda j,c: j>c), ('tied', lambda j,c: j==c))},
        'mean_absolute_within_case_motor_bias': {
            a: {field: [sum(v)/len(v) for v in dimensions] for field, dimensions in fields.items()}
            for a, fields in biases.items()},
        'gripper_mismatch_cases': {a: sum(r[a]['gripper_mismatches'] > 0 for r in case_rows) for a in arms},
        'gripper_wrong_teacher_margin': {
            a: {'statistics': stats([v['teacher_threshold_margin'] for v in wrong[a]]),
                'largest_margin_examples': sorted(wrong[a], key=lambda v: v['teacher_threshold_margin'], reverse=True)[:10]}
            for a in arms},
        'effective_model_input': {
            variant: {
                field: {'zero_cases': [r['case_id'] for r in inputs['cases'] if r['variants'][variant][field] == 0],
                        'statistics': stats([r['variants'][variant][field] for r in inputs['cases']])}
                for field in ('model_image_delta_rms', 'model_proprio_delta_rms')}
            for variant in p['variants'] if variant != 'base'},
    }
    assert len(case_rows) == 40
    write(out/'diagnostic_details.json', diagnostic)
    lines = ['# 40 visited states：同输入动作诊断', '',
             '原C80 final80 clip-only、joint与BF16；40个既有C8-student-visited states，四suite各10，原seed_index0。',
             '三组共享每state预处理后的输入tensor与采样seed。40×3×5=600端点，566次新查询、39份BF16 base缓存；训练0、rollout0。',
             'RMSE/MAE先逐case按前10步计算，再对40case等权；max_abs取跨case最大。原80-query held-out未改动。', '',
             '| Motor | clip normalized RMSE | joint normalized RMSE | clip command RMSE | joint command RMSE |',
             '|---|---:|---:|---:|---:|']
    for d, name in enumerate(p['motor_dimensions']):
        values = [endpoint[a]['motor'][field][d]['rmse'] for field in ('normalized_action','executed_command') for a in arms]
        lines.append('| %s | %s |' % (name, ' | '.join('%.8g' % v for v in values)))
    lines += ['', '| Base指标 | clip-only | joint |', '|---|---:|---:|']
    for label, values in (
        ('First10 motor RMSE', [endpoint[a]['motor']['normalized_action_overall_rmse_case_weighted'] for a in arms]),
        ('Gripper normalized RMSE', [endpoint[a]['gripper']['normalized_rmse'] for a in arms]),
        ('Gripper开合不一致 /400', [endpoint[a]['gripper']['command_sign_disagreement_count'] for a in arms]),
        ('至少一次开合不一致的states /40', [diagnostic['gripper_mismatch_cases'][a] for a in arms]),
        ('Chunk内切换位置不一致 /360', [endpoint[a]['gripper']['switch_timing_mismatch_count'] for a in arms])):
        lines.append('| %s | %s |' % (label, ' | '.join('%.8g' % v for v in values)))
    lines += ['', 'Gripper -1=open、+1=close、0=阈值零。以下confusion行是BF16、列是student，顺序[-1,0,+1]：', '']
    for arm in arms:
        lines.append('- %s: `%s`' % (arm, endpoint[arm]['gripper']['command_sign_confusion_counts']))
    lines += ['', 'Switch只比较step1–9各位置是否发生切换；不比较切换方向，也不包含上一chunk到step0的切换。', '',
              '| Motor | clip mean abs case bias | joint mean abs case bias | clip global signed bias | joint global signed bias |',
              '|---|---:|---:|---:|---:|']
    for d, name in enumerate(p['motor_dimensions']):
        values = [diagnostic['mean_absolute_within_case_motor_bias'][a]['normalized_action'][d] for a in arms]
        values += [endpoint[a]['motor']['normalized_action'][d]['signed_bias'] for a in arms]
        lines.append('| %s | %s |' % (name, ' | '.join('%.8g' % v for v in values)))
    lines += ['', 'Mean abs case bias先在每个state内平均10步误差，再取绝对值并平均states；避免不同state的正负偏置相消。它仍不是闭环漂移测量。', '',
              '| Suite | clip motor RMSE | joint motor RMSE | clip gripper mismatch /100 | joint gripper mismatch /100 |', '|---|---:|---:|---:|---:|']
    for suite, group in result['endpoint']['base']['by_suite_case_weighted'].items():
        ep = [group[a+'_endpoint_case_weighted'] for a in arms]
        values = [e['motor']['normalized_action_overall_rmse_case_weighted'] for e in ep]
        values += [e['gripper']['command_sign_disagreement_count'] for e in ep]
        lines.append('| %s | %s |' % (suite, ' | '.join('%.8g' % v for v in values)))
    for family, label in (('image', '两相机brightness ±1 uint8'), ('proprio', 'EEF xyz沿单位对角方向±5mm')):
        group = result['sensitivity']['families'][family]['overall']
        lines += ['', '## 局部方向响应：'+label, '',
                  '| Geometry case均值 | clip-only | joint |', '|---|---:|---:|']
        for key in ('student_rms','teacher_rms','error_rms','gain','cosine'):
            values = [group['arms'][a]['geometry'][key]['mean'] for a in arms]
            lines.append('| %s | %s |' % (key, ' | '.join('undefined' if v is None else '%.8g' % v for v in values)))
        lines += ['', '| Motor | BF16 derivative RMS | clip derivative error | joint derivative error | joint improved/worse/tied states |',
                  '|---|---:|---:|---:|---|']
        counts = group['joint_vs_clip_normalized_derivative_error_by_dimension']
        for d, name in enumerate(p['motor_dimensions']):
            cp, jt = [group['arms'][a]['central_directional_derivative']['normalized'][d] for a in arms]
            lines.append('| %s | %.8g | %.8g | %.8g | %s/%s/%s |' %
                         (name,cp['teacher_rms'],cp['teacher_error_rmse'],jt['teacher_error_rmse'],
                          counts['improved'][d],counts['worsened'][d],counts['tied'][d]))
        lines += ['', 'Geometry是60维motor方向响应的RMS、相对BF16误差、幅度gain及方向cosine，再对case等权；gain/cosine的defined case数见JSON。',
                  '不能跨image/proprio比较导数绝对大小，两者输入单位不同；响应越小不自动越好。']
    lines += ['', '## 证据范围', '',
              '这是40个既有baseline短程visited states的固定输入诊断；不是joint自身完整trajectory，也没有真实camera viewpoint扰动。',
              '两条预先固定的有限差分方向不能代表完整Jacobian或闭环稳定性，不能单独证明280-episode闭环差异的因果机制。',
              '更完整的逐维MAE/max、正负单侧响应、扰动gripper翻转、suite/case统计见comparison.json和diagnostic_details.json。',
              '有效BF16模型输入变化及饱和比例保留在input_perturbations.json；输出为decoded-BF16 VQ + fake A4，不是native INT4速度评测。']
    (out/'RESULTS.zh.md').write_text('\n'.join(lines)+'\n')
    write(out/'postprocess_summary.json', {'complete': True, 'pbs_jobid': os.environ['PBS_JOBID'],
          'source_gpu_job': summary['pbs_jobid'], 'new_model_queries':0, 'training_updates':0,
          'case_count':40, 'comparison_compact_bytes': (out/'comparison.compact.json').stat().st_size})


if __name__ == '__main__':
    main()
