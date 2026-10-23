"""Summarize completed direction probes on a CPU allocation."""
import json
import math
import os
from pathlib import Path
import statistics

from worker import guard, read, save


def main():
    guard(); root=Path(os.environ['NEW_ROOT']); p=read(root/'protocol.json'); runs={}
    import torch
    torch.set_num_threads(4)
    gradient_variation={}; baseline_cases={}
    for name,unit in p['units'].items():
        run=read(root/'results'/name/'summary.json')
        assert run['complete'] and run['checkpoint_arm']==unit['checkpoint_arm'] and run['loss']==unit['loss']
        assert run['parity_cases']==run['gradient_query_count']==80 and run['training_updates']==run['new_teacher_queries']==0
        assert run['parity_action_max_abs']==run['parity_future_max_abs']==0 and run['original_parameters_restored']
        assert len(run['direction_conditions'])==6 and all(v['queries']==80 for v in run['direction_conditions'])
        runs[name]=run
        gradient_state=torch.load(root/'results'/name/'gradient_state.pt',map_location='cpu',weights_only=True)
        assert gradient_state['protocol']==p and len(gradient_state['baseline'])==80
        assert [v['case_id'] for v in gradient_state['baseline']]==p['case_ids']
        baseline_cases[name]=gradient_state['baseline']
        norms=[v['per_query_surrogate_gradient_norm'] for v in gradient_state['baseline']]
        norm=run['full_gradient_norm']; squared_mean=sum(v*v for v in norms)/80
        assert norm>0 and all(math.isfinite(v) and v>=0 for v in norms)
        assert norm*norm<=squared_mean*(1+1e-5)
        gradient_variation[name]={'full_mean_gradient_norm':norm,'mean_query_gradient_norm':sum(norms)/80,
            'rms_query_gradient_norm':math.sqrt(squared_mean),
            'mean_norm_over_mean_query_norm':norm/(sum(norms)/80),
            'rms_query_deviation_over_mean_norm':math.sqrt(max(0,squared_mean-norm*norm))/norm,
            'scope':'80 frozen C80 per-query surrogate gradients at one fixed checkpoint; descriptive, not independent training replicates'}
        del gradient_state
    rows=[]
    for name,run in runs.items():
        for condition in run['direction_conditions']:
            label=condition['direction']+'_f'+str(condition['fraction']).replace('.','p')
            records=[read(root/'results'/name/'directions'/label/f"{v['case_id']}_0.json") for v in baseline_cases[name]]
            differences=[v['objective']-b['objective'] for v,b in zip(records,baseline_cases[name])]
            assert math.isclose(sum(differences)/80,condition['mean_delta'],rel_tol=1e-8,abs_tol=1e-12)
            suite_sums={suite:sum(d for d,b in zip(differences,baseline_cases[name]) if b['suite']==suite) for suite in condition['by_suite']}
            # Retain equal suite weights when omitting one C80 case.
            loco=[sum((total-d)/19 if suite==b['suite'] else total/20 for suite,total in suite_sums.items())/4
                  for d,b in zip(differences,baseline_cases[name])]
            rows.append({'unit':name,'checkpoint':run['checkpoint_arm'],'loss':run['loss'],**condition,
                'median_case_delta':statistics.median(differences),'loco_equal_suite_mean_delta_range':[min(loco),max(loco)],
                'mean_case_motor_rmse':sum(math.sqrt(v['action_mse']) for v in records)/80,
                'relative_change_percent':100*condition['mean_delta']/condition['baseline_objective']})
    output={'protocol':p['protocol'],'complete':True,'pbs_jobid':os.environ['PBS_JOBID'],
        'units':list(runs),'parity_cases':320,'gradient_queries':320,'direction_conditions':24,'direction_evaluation_queries':1920,
        'training_updates':0,'new_teacher_queries':0,'heldout_used':False,'rows':rows,
        'baseline':{name:{**run['baseline_metrics'],
                         'mean_case_motor_rmse':sum(math.sqrt(v['action_mse']) for v in baseline_cases[name])/80} for name,run in runs.items()},
        'component_gradients':{name:run['component_gradients'] for name,run in runs.items()},
        'gradient_variation':gradient_variation}
    source=Path(p['source_root']); offline=read(source/'results/c80_diagnosis/summary.json')
    assert offline['complete'] and not offline['uses_heldout']
    source_matrix=offline['objective_cross_matrix']; output['source_objective_cross_matrix']=source_matrix
    output['direction_decisions']={}
    for name,run in runs.items():
        known_mse=source_matrix['mse_nmse'][run['loss']]
        assert math.isclose(run['baseline_metrics']['objective'],source_matrix[run['checkpoint_arm']][run['loss']],rel_tol=1e-6)
        candidates=[{'direction':'no_step','fraction':0,'objective':run['baseline_metrics']['objective']}]+run['direction_conditions']
        best=min(candidates,key=lambda v:v['objective'])
        output['direction_decisions'][name]={'known_mse_objective_same_formula':known_mse,
            'best_including_no_step':{k:best[k] for k in ('direction','fraction','objective')},
            'best_below_known_mse':best['objective']<known_mse,
            'predicted_down_actual_up':sum(v['surrogate_linear_prediction']<0 and v['mean_delta']>0 for v in run['direction_conditions']),
            'scope':'C80-only finite counterfactual step probes; no deployment checkpoint saved, no held-out selection'}
    handles=read(root/'control/handles.json'); accounting={}
    for name,run in runs.items():
        attempts=[]
        for job in handles[name]['job_ids']:
            events=[]
            with (root/'artifacts'/job/'job.log').open(encoding='utf-8',errors='replace') as stream:
                for line in stream:
                    payload=line.partition('DIRECTION_PROGRESS ')[2]
                    if payload:
                        event=json.JSONDecoder().raw_decode(payload)[0]
                        assert event['pbs_jobid']==job
                        events.append(event)
            offset=next(v['new_model_queries'] for v in events if v['phase']=='parity_gradient')
            queries=max(v['new_model_queries'] for v in events)-offset
            assert queries>=0
            attempts.append({'pbs_jobid':job,'cached_gradient_query_counter_at_start':offset,'actual_model_queries_this_attempt':queries})
        expected=2*80+run['component_gradient_case_count']+6*80
        total=sum(v['actual_model_queries_this_attempt'] for v in attempts)
        assert total>=expected
        accounting[name]={'attempts':attempts,'completed_unique_model_queries':expected,'actual_model_queries':total,
            'recomputed_queries':total-expected,'new_teacher_queries':0,'training_updates':0}
    output['model_query_accounting']=accounting
    output['actual_model_queries']=sum(v['actual_model_queries'] for v in accounting.values())
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
        for ax,(name,run) in zip(axes.flat,runs.items()):
            for direction,label,color in [('saved_adam','Saved Adam moments','#bc4e30'),('fresh_adam','Fresh Adam moments','#007a87')]:
                conditions=sorted([v for v in rows if v['unit']==name and v['direction']==direction],key=lambda v:v['fraction'])
                ax.plot([0]+[v['fraction'] for v in conditions],[0]+[v['relative_change_percent'] for v in conditions],
                        marker='o',label=label,color=color)
            ax.axhline(0,color='#777777',linewidth=.8,linestyle='--'); ax.grid(alpha=.2)
            reference=100*(source_matrix['mse_nmse'][run['loss']]/run['baseline_metrics']['objective']-1)
            if abs(reference)>1e-5:
                ax.axhline(reference,color='#6b5689',linewidth=1,linestyle=':',label='Original MSE checkpoint')
            titles={'mse_dual':'MSE checkpoint / dual-KL objective','dual_dual':'Dual-KL checkpoint / dual-KL objective',
                    'latent_latent':'Latent-KL checkpoint / latent-KL objective','action_action':'Action-KL checkpoint / action-KL objective'}
            ax.set(title=titles[name],xlabel='Fraction of original step',ylabel='Full C80 objective change (%)')
            ax.set_xticks([0,.25,.5,1]);ax.legend(fontsize=8)
        fig.suptitle('Existing final80 checkpoints: measured full-C80 loss changes\n80 original calibration cases; no training updates; lower is better')
        for dpi in (125,100,85):
            fig.savefig(root/'results/real_loss_steps.png',dpi=dpi)
            if (root/'results/real_loss_steps.png').stat().st_size<=240*1024:break
        plt.close(fig)
        assert (root/'results/real_loss_steps.png').stat().st_size<=240*1024
        output['figure']='real_loss_steps.png'
    except ImportError as exc:
        output['figure_unavailable']=str(exc)
    header=['# KL parity 与真实方向诊断结果','',
        '全部结果使用原 C80/seed0 和冻结 teacher；没有训练更新或 held-out 调参。每条曲线来自一个已有 checkpoint 的完整 C80 平均 surrogate gradient，不等同于历史单 case 更新。','',
        '| Unit | 原 checkpoint | 当前目标 | Action parity max_abs | Future parity max_abs | Baseline objective |',
        '|---|---|---|---:|---:|---:|']
    for name,run in runs.items():
        header.append(f"| {name} | {run['checkpoint_arm']} | {run['loss']} | {run['parity_action_max_abs']} | {run['parity_future_max_abs']} | {run['baseline_metrics']['objective']:.9g} |")
    header+=['','| Unit | Adam moments | 步长比例 | 真实 C80 loss Δ% | Case中位Δ | Surrogate 一阶预测 Δ | 改善 cases |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        header.append(f"| {row['unit']} | {row['direction']} | {row['fraction']} | {row['relative_change_percent']:+.5f}% | {row['median_case_delta']:+.8g} | {row['surrogate_linear_prediction']:+.8g} | {row['improved_cases']}/80 |")
    header+=['','Case中位Δ及comparison.json中的equal-suite LOCO范围仅作固定C80的敏感性描述；LOCO每次移除一个case，并继续给四个suite各1/4权重，不作独立训练复现或显著性结论。']
    header+=['','## 完整 C80 梯度的 case 间变化','',
        '`||mean g|| / mean ||g_i||` 越小，表示该固定 checkpoint 上不同 case 的梯度抵消越强。后一列为 `sqrt(mean ||g_i-mean g||^2) / ||mean g||`，只描述这80个case，不代表跨训练复现的不确定性。','',
        '| Unit | 完整均值梯度范数 | 单case范数均值 | 均值范数/单case范数均值 | Case偏离RMS/均值范数 |',
        '|---|---:|---:|---:|---:|']
    for name,v in gradient_variation.items():
        header.append(f"| {name} | {v['full_mean_gradient_norm']:.8g} | {v['mean_query_gradient_norm']:.8g} | {v['mean_norm_over_mean_query_norm']:.5f} | {v['rms_query_deviation_over_mean_norm']:.5f} |")
    header+=['','## 是否超过已知更好的原 MSE 解','',
        '每行使用该unit的同一种目标公式，候选中包含不更新。最佳值只是已测试条件的最小值，不是最优步长或收敛证明。','',
        '| Unit | 原MSE的同公式目标 | 最佳候选 | 最佳目标 | 低于原MSE | 预测下降但实际上升 |',
        '|---|---:|---|---:|---|---:|']
    for name,v in output['direction_decisions'].items():
        b=v['best_including_no_step']
        header.append(f"| {name} | {v['known_mse_objective_same_formula']:.9g} | {b['direction']} × {b['fraction']} | {b['objective']:.9g} | {v['best_below_known_mse']} | {v['predicted_down_actual_up']}/6 |")
    header+=['','## 续跑与查询计数','',
        '每次attempt从其首次parity_gradient事件中的缓存梯度计数开始，按job.log中实际增加的query计数核算；不重复计算续跑复用的梯度前缀或已保存端点。','',
        '| Unit | Attempts | 完整唯一model queries | 实际model queries | 重算queries |',
        '|---|---:|---:|---:|---:|---:|']
    for name,v in accounting.items():
        header.append(f"| {name} | {len(v['attempts'])} | {v['completed_unique_model_queries']} | {v['actual_model_queries']} | {v['recomputed_queries']} |")
    header+=['','分项梯度仅来自预先指定的4个C80case；范数已包含固定alpha和lambda，None梯度按未连接处理，夹角不代表全C80聚合梯度夹角。','',
             '| Unit | Case | 参数组 | Action norm | Latent norm | Cosine |','|---|---:|---|---:|---:|---:|']
    for name,run in runs.items():
        for case in run['component_gradients']:
            for group in ('ratio','channel'):
                v=case['summary'][group]
                header.append(f"| {name} | {case['case_id']} | {group} | {v['norm_a']:.7g} | {v['norm_z']:.7g} | {v['cosine']:+.5f} |")
    if output.get('figure'):header+=['','![Full C80 real loss steps](real_loss_steps.png)']
    header+=['','证据边界：诊断没有保存训练后的模型；每个扰动后恢复原参数。负方向预测与真实forward不一致只能指示当前有限步/离散surrogate问题，不能证明不存在更小有效步长。完整C80下降也不证明held-out或闭环改善。']
    (root/'results/ANALYSIS.zh.md').write_text('\n'.join(header)+'\n',encoding='utf-8')
    save(root/'results/comparison.json',output)
    save(root/'results/aggregate/summary.json',{k:output[k] for k in ('protocol','complete','pbs_jobid','parity_cases','gradient_queries','direction_conditions','direction_evaluation_queries','training_updates','new_teacher_queries','heldout_used')})
    print('KL_DIRECTION_AGGREGATE_COMPLETE',flush=True)


if __name__=='__main__':main()
