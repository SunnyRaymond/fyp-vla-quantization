"""Independent NumPy analysis of the frozen reference-branch screen.

Call only in a verified CCDS CPU allocation. No model or producer import.
"""
import numpy as np

IDS = (1035, 1534, 1158, 203, 1837, 1095)


def replay(raw):
    checks = {}
    result = {'checks': checks, 'decision': 'implementation_inconclusive', 'rows': []}
    expected = {
        'pred_visual': ((6, 5, 196, 384), 'float32'),
        'goal_visual': ((6, 5, 196, 384), 'float32'),
        'current_visual': ((6, 5, 196, 384), 'float32'),
        'fp_noop_pred': ((6, 2, 196, 384), 'float32'),
        'fp_noop_goal': ((6, 2, 196, 384), 'float32'),
        'fp_noop_current': ((6, 2, 196, 384), 'float32'),
        'predictor_input': ((6, 5, 196, 404), 'float32'),
        'completed': ((6, 5), 'bool'),
        'trajectory_ids': ((6,), 'int64'),
        'valid_indices': ((6,), 'int64'),
    }
    checks['required_arrays'] = set(expected).union({'arm_names', 'schema'}).issubset(raw.keys())
    if not checks['required_arrays']:
        return result
    arm_names = np.asarray(raw['arm_names'])
    schema = np.asarray(raw['schema'])
    checks['arm_order'] = bool(arm_names.shape == (5,) and arm_names.tolist() ==
        ['FP32', 'encoder_W4_RTN', 'encoder_W4_SR0', 'encoder_W4_SR1', 'encoder_W4_SR2'])
    checks['raw_schema'] = bool(schema.shape == () and schema.item() == 'reference-branch-coupling-raw-v1')
    a = {key: np.asarray(raw[key]) for key in expected}
    checks['array_shapes'] = all(a[k].shape == shape for k, (shape, _) in expected.items())
    checks['array_dtypes'] = all(a[k].dtype == np.dtype(dtype) for k, (_, dtype) in expected.items())
    if not all(checks.values()):
        return result
    checks['complete'] = bool(a['completed'].all())
    if not checks['complete']:
        result['decision'] = 'inconclusive_budget'
        return result
    checks['finite'] = all(bool(np.isfinite(v).all()) for v in a.values())
    checks['trajectories'] = bool(np.array_equal(a['trajectory_ids'], IDS))
    checks['indices'] = bool(np.array_equal(a['valid_indices'], np.arange(124, 130)))
    checks['actual_predictor_visual'] = bool(np.array_equal(a['predictor_input'][..., :384], a['current_visual']))
    tail = a['predictor_input'][..., 384:]
    checks['nonvisual_input_unchanged'] = bool(np.array_equal(tail, np.broadcast_to(tail[:, :1], tail.shape)))
    for name, value in [('pred', 'pred_visual'), ('goal', 'goal_visual'), ('current', 'current_visual')]:
        checks['fp_noop_' + name] = bool(np.array_equal(
            a['fp_noop_' + name], np.broadcast_to(a[value][:, :1], a['fp_noop_' + name].shape)))
    if not all(checks.values()):
        return result
    diagonal = np.eye(3, dtype=bool)
    off = ~diagonal
    def mse(v):
        return float(np.mean(np.square(v), dtype=np.float64))
    def summarize(v):
        return float(np.mean(v[diagonal])), float(np.mean(v[off]))
    bound = positive = 0
    algebra_ok = marginals_ok = True
    for i in range(6):
        y = a['pred_visual'][i].astype(np.float64)
        g = a['goal_visual'][i].astype(np.float64)
        r_fp = y[0] - g[0]
        l_fp = mse(r_fp)
        dy = y[2:] - y[0]
        dg = g[2:] - g[0]
        aa = np.empty((3, 3), dtype=np.float64)
        bb = np.empty_like(aa)
        ll = np.empty_like(aa)
        cross = np.empty_like(aa)
        y_norm = np.array([mse(v) for v in dy])
        g_norm = np.array([mse(v) for v in dg])
        for d in range(3):
            for e in range(3):
                r = y[d+2] - g[e+2]
                aa[d, e] = mse(r - r_fp)
                ll[d, e] = mse(r)
                bb[d, e] = (ll[d, e] - l_fp) ** 2
                cross[d, e] = float(np.mean(dy[d] * dg[e], dtype=np.float64))
        expanded = y_norm[:, None] + g_norm[None, :] - 2.0 * cross
        algebra_ok &= bool(np.allclose(aa, expanded, atol=1e-12, rtol=1e-9))
        y_matrix = np.broadcast_to(y_norm[:, None], (3, 3))
        g_matrix = np.broadcast_to(g_norm[None, :], (3, 3))
        ys, yd = summarize(y_matrix)
        gs, gd = summarize(g_matrix)
        marginals_ok &= bool(np.isclose(ys, yd, atol=1e-12, rtol=1e-10)
                             and np.isclose(gs, gd, atol=1e-12, rtol=1e-10))
        sa, da = summarize(aa)
        sb, db = summarize(bb)
        binding = bool(da > 1e-12 and db > 1e-12)
        ga = (da - sa) / da if da > 1e-12 else None
        gb = (db - sb) / db if db > 1e-12 else None
        joint = bool(binding and ga >= 0.10 and gb >= 0.10)
        bound += int(binding)
        positive += int(joint)
        result['rows'].append({
            'state_index': i, 'trajectory_id': IDS[i], 'binding': binding, 'joint_positive': joint,
            'A_shared': sa, 'A_different': da, 'B_shared': sb, 'B_different': db,
            'gain_A': ga, 'gain_B': gb, 'FP_visual_objective': l_fp,
            'RTN_residual_error': mse((y[1]-g[1])-r_fp),
            'RTN_objective_error': (mse(y[1]-g[1])-l_fp)**2,
            'A_matrix': aa.tolist(), 'B_matrix': bb.tolist(), 'L_matrix': ll.tolist(),
            'cross_inner_product_matrix': cross.tolist(),
            'propagated_error_marginals': [ys, yd], 'goal_error_marginals': [gs, gd],
        })
    checks['MSE_decomposition'] = bool(algebra_ok)
    checks['matched_branch_error_marginals'] = bool(marginals_ok)
    checks['engineering_pass'] = all(checks.values())
    result['binding_count'] = bound
    result['positive_count'] = positive
    result['thresholds'] = {'binding_A_B': 1e-12, 'relative_gain_A_B': 0.10,
                            'minimum_binding_states': 4, 'joint_positive_states': 4}
    if not checks['engineering_pass']:
        return result
    result['decision'] = ('inconclusive_binding' if bound < 4 else
                          'scope_limited_preliminary_go' if positive >= 4 else 'mechanism_no_go')
    return result
