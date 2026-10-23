"""CPU postprocess of the predeclared C8 mechanism probes, without model loading."""
import copy
import argparse
import math
import os
from pathlib import Path
import statistics as st
from anchor_selfcheck import require_allocation
from prepare import read,save


def compare(left,right):
    differences=[a-b for a,b in zip(left,right)]
    return {'mean_delta':st.mean(differences),'median_delta':st.median(differences),
            'better':sum(v<0 for v in differences),'worse':sum(v>0 for v in differences),
            'ties':sum(v==0 for v in differences),'differences':differences}


def local_projection(alignments):
    zk=alignments['weighted_nmse__weighted_kl']['all']
    coefficient=zk['dot']/zk['left_norm']**2
    norm=math.sqrt(max(0.,zk['right_norm']**2-zk['dot']**2/zk['left_norm']**2))
    ak=alignments['action__weighted_kl']['all'];az=alignments['action__weighted_nmse']['all']
    numerator=ak['dot']-coefficient*az['dot']
    return {'coefficient':coefficient,'action_cosine_orthogonal_kl':numerator/(ak['left_norm']*norm) if norm else None}


def optimizer_geometry(parent,config,point,panels,source):
    """Posthoc CPU-only decomposition prompted by zero deployed gain changes."""
    import torch
    from mechanism_math import mean_gradients,mix,adam_delta,dot,group_alignment
    torch.set_num_threads(4)
    records=[torch.load(source/'gradients'/point/f'{c}.pt',map_location='cpu',weights_only=True) for c in config['case_ids']]
    names=records[0]['parameter_names']
    for case,record in zip(config['case_ids'],records):
        if (record['case_id']!=case or record['point']!=point or record['parameter_names']!=names
                or record['binding']['protocol']!=config['protocol']):raise ValueError('Cached gradient binding differs')
    gradients=mean_gradients(records);del records
    saved=None
    if point=='initial':
        values=[]
        for index,name in enumerate(names):
            bank=torch.load(Path(parent['calibration']['bank_directory'])/f'{index:04d}.pt',map_location='cpu',weights_only=True)
            if bank['module']!=name or bank['module_index']!=index or bank['cohort']!='c8':raise ValueError('Original C8 bank reconstruction differs')
            encoded=bank['encoded'];scale=encoded['row_scales'].float()
            values.extend([encoded['books'].float(),scale.clamp_min(torch.finfo(torch.float32).tiny).log()])
            del bank,encoded,scale
    else:
        saved=torch.load(source.parent/'anchor_kl/checkpoint.pt',map_location='cpu',weights_only=True)
        if saved['protocol']!=parent or saved['updates']!=8:raise ValueError('Final KL optimizer snapshot differs')
        values=[saved['module_masters'][name][key] for name in names for key in ('books','log_scales')]
    proposals={}
    for mode in ['fresh']+(['saved'] if saved else []):
        for fraction in config['beta_fractions']:
            proposals[(mode,fraction)]=adam_delta(torch,values,mix(gradients,fraction),config['learning_rate'],
                saved['optimizer'] if mode=='saved' else None)
    pure=adam_delta(torch,values,gradients['weighted_kl'],config['learning_rate'])
    factor=math.sqrt(dot(proposals[('fresh',0.)],proposals[('fresh',0.)])/dot(pure,pure))
    pure=[v*factor for v in pure]
    result={'posthoc':True,'new_model_queries':0,'panels':[],
        'schema':1,'protocol':config['protocol'],'point':point,'case_ids':config['case_ids'],
        'initial_reconstruction_note':'CPU log of original BF16 gains may differ from original GPU log in final FP32 bits; norm and g-dot-delta reproduction discrepancies are recorded.',
        'step_alignments':{mode+'_beta'+str(fraction)+'__beta0':group_alignment(proposals[(mode,fraction)],proposals[(mode,0.)])
            for mode in ['fresh']+(['saved'] if saved else []) for fraction in (.1,1.)}}
    for panel in panels:
        delta=pure if panel['beta_fraction'] is None else proposals[(panel['optimizer_state'],panel['beta_fraction'])]
        delta=[v*panel['step_fraction'] for v in delta]
        by_group={group:{term:dot(gradient,delta,indices) for term,gradient in gradients.items()}
                  for group,indices in [('books',range(0,len(values),2)),('log_row_gains',range(1,len(values),2))]}
        reproduced={term:sum(v[term] for v in by_group.values()) for term in gradients}
        errors={term:abs(reproduced[term]-expected)/max(abs(expected),1e-30)
                for term,expected in panel['predicted_weighted_component_delta'].items()}
        result['panels'].append({'label':panel['label'],'weighted_prediction_by_group':by_group,
            'prediction_relative_errors':errors,'delta_norm_relative_error':abs(math.sqrt(dot(delta,delta))-panel['parameter_delta_norm'])/panel['parameter_delta_norm'],
            'source_stored_coordinates_changed':panel['stored_coordinates_changed'],
            'actual_base_delta':panel['delta']['base_objective']})
    result['max_prediction_relative_error']=max(v for panel in result['panels'] for v in panel['prediction_relative_errors'].values())
    result['max_delta_norm_relative_error']=max(panel['delta_norm_relative_error'] for panel in result['panels'])
    result['reconstruction_passed']=result['max_prediction_relative_error']<1e-3 and result['max_delta_norm_relative_error']<1e-3
    return result


def preview_initial():
    require_allocation()
    parent=read(Path(__file__).with_name('protocol.json'));config=read(Path(__file__).with_name('mechanism_protocol.json'))
    root=Path(parent['remote_root']);source=root/'results/mechanism'
    panels=read(source/'initial/panels.json')
    if len(panels)!=8 or any(sorted(r['case_id'] for r in panel['cases'])!=sorted(config['case_ids']) for panel in panels):
        raise ValueError('Preview requires all eight initial panels on the full C8')
    result=optimizer_geometry(parent,config,'initial',panels,source)
    save(root/'results/mechanism_analysis/initial_optimizer_geometry.json',result)
    save(root/'results/mechanism_initial_opt/summary.json',{
        'protocol':config['protocol'],'unit':'mechanism_initial_opt','pbs_jobid':os.environ['PBS_JOBID'],
        'complete':True,'new_model_queries':0,'new_backwards':0,'training_updates':0,'heldout_used':False,
        'result':result})
    print('MECHANISM_INITIAL_OPT_COMPLETE',flush=True)


def main():
    require_allocation()
    parent=read(Path(__file__).with_name('protocol.json'))
    config=read(Path(__file__).with_name('mechanism_protocol.json'))
    root=Path(parent['remote_root']);source=root/'results/mechanism'
    raw=read(source/'summary.json');handles=read(root/'control/mechanism_handles.json')
    handle=handles['mechanism'];artifact=root/'artifacts'/handle['job_id']
    if (not raw['complete'] or raw['completed_pbs_jobid']!=handle['job_id']
            or raw['gradient_records']!=16 or raw['parameter_probe_panels']!=22
            or raw['parameter_probe_queries']!=176 or raw['reader_record_count']!=8
            or (artifact/'exit_code.txt').read_text().strip()!='0' or not (artifact/'PHASE_COMPLETE').is_file()):
        raise ValueError('Complete source job, artifacts and declared C8 coverage required')
    if raw['config']!=config:raise ValueError('Frozen mechanism configuration differs')
    lam=parent['training']['rho']*parent['training']['normalization_c']
    beta=read(root/'results/prepare/summary.json')['beta']
    initial_rows={r['case_id']:r for r in read(root/'results/prepare/summary.json')['initial_rows']}
    old=read(root/'results/diagnosis_v2/summary.json')
    result={'protocol':config['protocol'],'unit':'mechanism_analysis','pbs_jobid':os.environ['PBS_JOBID'],
            'source_job_id':handle['job_id'],'complete':False,'new_model_queries':0,'training_updates':0,
            'heldout_used':False,'parameter_gradients':{},'weight_comparisons':[],
            'finite_step_predictions':[],'pure_kl_probes':[],'reader':{},'optimizer_geometry':{}}
    lines=['# C8 anchor KL 权重、优化与方向：完整机制诊断','',
        '本报告仅覆盖固定 C8、两个参数点和预先声明的小步干预。沿用 teacher 和缓存端点，原 trained baseline 保留既有结果。各 case 固定一个 sampler seed；两个参数点及同一输入的多次 probe 均不是独立训练重复。','',
        '## 参数梯度','',
        '| 参数点 | base 与 KL cosine | KL/base 梯度 norm | action 与 KL cosine | action 与 NMSE cosine | KL 跨 case 负 cosine pairs / 28 |',
        '|---|---:|---:|---:|---:|---:|']
    for point in config['points']:
        geometry=read(source/point/'gradient_geometry.json');panels=read(source/point/'panels.json')
        mean=geometry['mean_gradient_alignments'];basekl=mean['base__weighted_kl']['all']
        result['parameter_gradients'][point]={'geometry':geometry,'kl_to_base_norm':basekl['right_norm']/basekl['left_norm']}
        zk=mean['weighted_nmse__weighted_kl']['all']
        coefficient=zk['dot']/zk['left_norm']**2
        orthogonal_norm=math.sqrt(max(0.,zk['right_norm']**2-zk['dot']**2/zk['left_norm']**2))
        action_orthogonal_dot=mean['action__weighted_kl']['all']['dot']-coefficient*mean['action__weighted_nmse']['all']['dot']
        result['parameter_gradients'][point]['kl_gradient_projection_on_nmse']={
            'coefficient':coefficient,
            'local_projected_effective_rho':parent['training']['rho']*(1+coefficient),
            'orthogonal_kl_norm_over_base_norm':orthogonal_norm/basekl['left_norm'],
            'action_dot_orthogonal_kl':action_orthogonal_dot,
            'action_cosine_orthogonal_kl':action_orthogonal_dot/(mean['action__weighted_kl']['all']['left_norm']*orthogonal_norm) if orthogonal_norm else None,
            'base_cosine_orthogonal_kl':action_orthogonal_dot/(basekl['left_norm']*orthogonal_norm) if orthogonal_norm else None,
            'scope':'Local gradient projection only, not equivalence of the two losses or a recommended rho.'}
        old_phase='initial' if point=='initial' else 'final'
        output_cosines={r['case_id']:r['future_gradient_cosine'] for r in old['latent_diagnostics'][old_phase]}
        result['parameter_gradients'][point]['paired_output_parameter_cosines']=[{
            'case_id':r['case_id'],'output_nmse_kl_cosine':output_cosines[r['case_id']],
            'parameter_nmse_kl_cosine':r['alignments']['weighted_nmse__weighted_kl']['all']['cosine'],
            'within_case_kl_projection':local_projection(r['alignments'])}
            for r in geometry['per_case']]
        lines.append(f"| {point} | {basekl['cosine']:.6g} | {basekl['right_norm']/basekl['left_norm']:.6g} | {mean['action__weighted_kl']['all']['cosine']:.6g} | {mean['action__weighted_nmse']['all']['cosine']:.6g} | {geometry['cross_case']['weighted_kl']['negative_pairs']} / 28 |")
        lookup={(r['optimizer_state'],r['step_fraction'],r['beta_fraction']):r for r in panels if r['beta_fraction'] is not None}
        modes=['fresh']+(['saved'] if point=='final_kl' else [])
        for mode in modes:
            for fraction in config['step_fractions']:
                arms={f:lookup[(mode,fraction,f)] for f in config['beta_fractions']}
                rows={f:{r['case_id']:r for r in arm['cases']} for f,arm in arms.items()}
                contrast={'point':point,'optimizer':mode,'step_fraction':fraction,'metrics':{}}
                for metric in ('action_mse','base_objective','latent_nmse','latent_anchor_kl'):
                    contrast['metrics'][metric]={str(f):compare(
                        [rows[f][c][metric] for c in config['case_ids']],
                        [rows[1.0][c][metric] for c in config['case_ids']]) for f in (0.0,0.1)}
                result['weight_comparisons'].append(contrast)
        for panel in panels:
            prediction=panel['predicted_weighted_component_delta'];change=panel['delta']
            references=initial_rows if point=='initial' else {
                c:read(root/'results/anchor_kl/calibration/queries'/f'{c}_0.json') for c in config['case_ids']}
            cases={r['case_id']:r for r in panel['cases']}
            actual_case_comparisons={
                'action':compare([cases[c]['action_mse'] for c in config['case_ids']],
                                 [references[c]['action_mse'] for c in config['case_ids']]),
                'base':compare([cases[c]['base_objective'] for c in config['case_ids']],
                               [references[c]['action_mse']+lam*references[c]['latent_nmse'] for c in config['case_ids']])}
            measures={'action':(prediction['action'],change['action_mse']),
                      'base':(prediction['action']+prediction['weighted_nmse'],change['base_objective']),
                      'kl':(prediction['weighted_kl'],beta*change['latent_anchor_kl'])}
            result['finite_step_predictions'].append({'point':point,'panel':panel['label'],
                'actual_vs_point_by_case':actual_case_comparisons,
                'components':{m:{'predicted':a,'actual':b,'predicted_descent_but_actual_ascent':a<0 and b>0}
                              for m,(a,b) in measures.items()},
                'stored_coordinates_changed':panel['stored_coordinates_changed']})
            if panel['beta_fraction'] is None:
                result['pure_kl_probes'].append({'point':point,'step_fraction':panel['step_fraction'],
                    'actual_kl_delta':change['latent_anchor_kl'],'actual_action_delta':change['action_mse'],
                    'actual_base_delta':change['base_objective'],
                    'kl_down_action_up':change['latent_anchor_kl']<0 and change['action_mse']>0})
        print('MECHANISM_CPU_OPTIMIZER',point,flush=True)
        cached=root/'results/mechanism_analysis'/f'{point}_optimizer_geometry.json'
        opt=read(cached) if cached.is_file() else None
        if not opt or (opt.get('schema'),opt.get('protocol'),opt.get('point'),opt.get('case_ids'))!=(1,config['protocol'],point,config['case_ids']):
            opt=optimizer_geometry(parent,config,point,panels,source);save(cached,opt)
        result['optimizer_geometry'][point]={k:v for k,v in opt.items() if k!='panels'}
        for panel in opt['panels']:
            groups=panel['weighted_prediction_by_group']
            result['optimizer_geometry'][point].setdefault('base_prediction_by_group',[]).append({
                'label':panel['label'],
                'books':groups['books']['action']+groups['books']['weighted_nmse'],
                'log_row_gains':groups['log_row_gains']['action']+groups['log_row_gains']['weighted_nmse'],
                'deployed_row_gains_changed':panel['source_stored_coordinates_changed']['row_gains']['changed'],
                'actual_base_delta':panel['actual_base_delta']})
    lines+=['','完整逐 case、books 与 log-row-gains 的梯度几何保存在 JSON。跨 case 负方向是共享参数下的局部干扰证据，不单独证明历史遗忘；STE 梯度也不保证实际量化 forward 下降。','',
        '将加权 KL 参数梯度写成 a × 加权 NMSE 梯度 + 正交部分，JSON 报告 a、对应的局部有效 rho，以及正交部分与 action 梯度的关系。这只是同一点的 surrogate 梯度分解，不等于两个 loss 全局等价或 Adam 实际更新等价。全 C8 平均梯度的正交部分同时包含逐 case 方向差异与隐含样本权重变化，不能全归因于新的结构方向；JSON 另给逐 case 投影。另逐 case 对齐输出 latent 与参数空间的 cosine，避免把聚合差异误认为模型反传的作用。','',
        '## 减小 beta 的真实 C8 效果','',
        '表中差值为 beta=0.1 倍减去原 beta，均为同一点、同 Adam 状态、同一步长；负值表示较小 beta 更好。','',
        '| 点 | Adam | 步长比例 | Action MSE 平均差 | Action 改善/变差 | 原 MSE+NMSE 平均差 |',
        '|---|---|---:|---:|---:|---:|']
    for c in result['weight_comparisons']:
        a=c['metrics']['action_mse']['0.1'];b=c['metrics']['base_objective']['0.1']
        lines.append(f"| {c['point']} | {c['optimizer']} | {c['step_fraction']} | {a['mean_delta']:+.7g} | {a['better']}/{a['worse']} | {b['mean_delta']:+.7g} |")
    lines+=['','beta=0 对照、配对中位数和全部 8-case 差值保留在 JSON；不将本组条件用作新 checkpoint 选择。','',
        '## 真实 finite-step 与纯 KL 方向','',
        '| 点 | 步长比例 | 纯 KL 更新的实际 KL 差 | Action MSE 差 | KL 降、action 升 |',
        '|---|---:|---:|---:|---|']
    for c in result['pure_kl_probes']:
        lines.append(f"| {c['point']} | {c['step_fraction']} | {c['actual_kl_delta']:+.7g} | {c['actual_action_delta']:+.7g} | {c['kl_down_action_up']} |")
    result['finite_step_disagreement_counts']={component:sum(r['components'][component]['predicted_descent_but_actual_ascent'] for r in result['finite_step_predictions'])
        for component in ('action','base','kl')}
    lines+=['','## 部署量化与 Adam 几何的补充分析','',
        '这一部分在观察到初始化 probe 的 BF16 row gains 无变化后追加，只重读既有梯度、原 C8 banks 和 checkpoint。它不改变任何 GPU 条件，不新增 forward。CPU 重建的 norm/g-dot-delta 与 GPU probe 保存值的偏差记录在 JSON。','',
        '| 点 | 条件 | Base 预测差：books | Base 预测差：row gains | 部署 gain 改变数 | 实际 base 差 |',
        '|---|---|---:|---:|---:|---:|']
    for point,opt in result['optimizer_geometry'].items():
        for r in opt['base_prediction_by_group']:
            if 'beta1.0' in r['label']:
                lines.append(f"| {point} | {r['label']} | {r['books']:+.7g} | {r['log_row_gains']:+.7g} | {r['deployed_row_gains_changed']} | {r['actual_base_delta']:+.7g} |")
    lines+=['','非零 FP32 master delta 可以在 BF16 存储后完全消失。按参数组分解的一阶预测用于定位这种落差；它仍不能单独把全部真实-loss 变化归因于 rounding，因为 books 更新和 A4 的 STE 误差也同时存在。','']
    lines+=['',f"22 个参数 probe panels 中，surrogate 预测下降但真实均值上升的计数：{result['finite_step_disagreement_counts']}。这些条件相关，不是 22 次独立实验。步长、量化 rounding 与 STE 方向近似可能共同影响结果，不能仅凭符号不一致认定 autograd 实现错误。",'',
        '各 panel 相对原参数点的逐 case action/base 改善数、配对中位数和差值也保存在 JSON，不以单个均值代表全部 C8。','',
        '## 等 NMSE 的 latent 方向干预','']
    reader=raw['reader_records']
    old_ranks={r['case_id']:r['local_anchor_rank']['mean']*196 for r in old['latent_diagnostics']['final']}
    for r in reader:
        null=r['reader_gradient']['reader_gradient_raw_null_energy_fraction'];rank=round(old_ranks[r['case_id']]);dimension=48*2*14*28
        r['reader_gradient']['visible_dimensions']=rank;r['reader_gradient']['full_dimensions']=dimension
        r['reader_gradient']['isotropic_null_dimension_fraction']=1-rank/dimension
        r['reader_gradient']['visible_to_null_gradient_energy_per_dimension']=((1-null)/rank)/(null/(dimension-rank)) if null else None
    keys=('original','repair_visible','repair_null')
    means={key:{m:st.mean(r['outputs'][key][m] for r in reader) for m in reader[0]['outputs'][key]} for key in keys}
    contrasts={m:compare([r['outputs']['repair_visible'][m] for r in reader],
                         [r['outputs']['repair_null'][m] for r in reader])
               for m in ('action_mse','first10motor_rmse','full32motor_rmse','first10gripper_rmse','latent_nmse','latent_anchor_kl')}
    result['reader']={'mean_metrics':means,'repair_visible_minus_repair_null':contrasts,'cases':reader}
    lines+=['两种修复从同一个保存的 candidate future 出发，实际 BF16 注入后匹配 raw NMSE；同一原 BF16 reader、同一 action noise，teacher-future reader 重放通过。','',
        '| 修复 | Future NMSE | Anchor KL | Reader first10 motor RMSE | Reader action MSE |',
        '|---|---:|---:|---:|---:|']
    for key,v in means.items():lines.append(f"| {key} | {v['latent_nmse']:.7g} | {v['latent_anchor_kl']:.7g} | {v['first10motor_rmse']:.7g} | {v['action_mse']:.7g} |")
    c=contrasts['action_mse']
    lines+=['',f"修复可见部分 − 修复 null 部分，reader action MSE 平均差 {c['mean_delta']:+.7g}，中位数 {c['median_delta']:+.7g}，前者更好 {c['better']}/8、后者更好 {c['worse']}/8。",'',
        'Nullspace 维数很大，不能只看其包含的梯度能量比例：完整 future 有 37632 个坐标，KL 可见空间约 1248 维。JSON 同时报梯度能量按维数归一化后的集中程度。修复比较控制最终 NMSE，但两种 repair 的位移长度未必相同；它是误差结构干预，不是等幅梯度方向测试。','',
        '## 完成与边界','',
        f"源 GPU 作业 {handle['job_id']}；CPU 汇总作业 {os.environ['PBS_JOBID']}。16/16 参数梯度 records、22/22 参数 panels（176 case queries）、8/8 reader records（32 reader queries）。",
        f"累计 student full queries={raw['new_student_full_queries']}，reader-only queries={raw['new_reader_queries']}，参数 reverse passes={raw['new_parameter_reverse_passes']}，reader reverse passes={raw['new_reader_reverse_passes']}。内部 checkpoint recomputation 不作为额外逻辑 query。",
        '原 trained baseline 0 新 queries、0 新 teacher full queries、0 持久 training updates、0 closed-loop episodes。没有通过上述数据证明独立泛化或闭环优劣。权重与方向证据可能同时成立；最终解释需结合这几张表，不把任一单个 cosine 或平均数当作机制证明。']
    result['complete']=True
    out=root/'results/mechanism_analysis';save(out/'details.json',result)
    compact=copy.deepcopy(result)
    for value in compact['parameter_gradients'].values():value['geometry'].pop('per_case')
    for comparison in compact['weight_comparisons']:
        for contrasts in comparison['metrics'].values():
            for value in contrasts.values():value.pop('differences')
    for panel in compact['finite_step_predictions']:
        for value in panel['actual_vs_point_by_case'].values():value.pop('differences')
    save(out/'summary.json',compact)
    report=root/'results/MECHANISM_ANALYSIS.zh.md'
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('MECHANISM_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preview-initial',action='store_true')
    if parser.parse_args().preview_initial:preview_initial()
    else:main()
