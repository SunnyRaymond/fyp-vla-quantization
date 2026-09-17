"""Independent NumPy replay, called only by a guarded compute-node verifier.

No model inference: mu_values are bound FP scorer outputs, not CPU recomputation.
This module is a library and intentionally has no local CLI entrypoint.
"""


def replay(data, np):
    shapes = {'actions': (8,3,3,512,1), 'values': (8,3,512),
              'elite_indices': (8,3,64), 'weights': (8,3,64),
              'mu': (8,3,3,1), 'std': (8,3,3,1),
              'mu_values': (8,3), 'masses': (8,3), 'counts': (8,3)}
    for name, shape in shapes.items():
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError(f'Invalid finite array shape: {name}')
    actions = data['actions'].astype(np.float64)
    values = data['values'].astype(np.float64)
    if (np.abs(actions)>1+1e-7).any():
        raise ValueError('Actions violate frozen clamp/squash bound')
    checks = {'common_candidates_exact': bool(np.array_equal(actions[:,0,:,24:], actions[:,1,:,24:])
              and np.array_equal(actions[:,0,:,24:], actions[:,2,:,24:])),
              'common_scores_close': bool(np.allclose(values[:,0,24:],values[:,1,24:],atol=1e-5,rtol=1e-6)
              and np.allclose(values[:,0,24:],values[:,2,24:],atol=1e-5,rtol=1e-6))}
    top_set_ok = weights_ok = mu_ok = std_ok = masses_ok = counts_ok = True
    masses = np.zeros((8,3),dtype=np.float64)
    counts = np.zeros((8,3),dtype=np.int64)
    cutoff_ties = np.zeros((8,3),dtype=np.bool_)
    for i in range(8):
        for arm in range(3):
            idxraw = data['elite_indices'][i,arm]
            idx = idxraw.astype(np.int64)
            if not np.array_equal(idxraw,idx) or len(set(idx.tolist()))!=64 or (idx<0).any() or (idx>=512).any():
                raise ValueError('Elite indices are not 64 distinct valid integers')
            sorted_values = np.sort(values[i,arm])[::-1]
            cutoff_ties[i,arm] = sorted_values[63] == sorted_values[64]
            # Tied order is irrelevant for verification; require valid top-64 values.
            top_set_ok &= bool(np.array_equal(np.sort(values[i,arm,idx])[::-1],sorted_values[:64]))
            v = values[i,arm,idx]
            w = np.exp(.5*(v-v.max()))
            w /= w.sum()
            selected = actions[i,arm][:,idx,:]
            mu = (w[None,:,None]*selected).sum(axis=1)/(w.sum()+1e-9)
            std = np.sqrt((w[None,:,None]*np.square(selected-mu[:,None,:])).sum(axis=1)/(w.sum()+1e-9))
            std = np.clip(std,.05,2.)
            masses[i,arm] = w[idx<24].sum()
            counts[i,arm] = (idx<24).sum()
            weights_ok &= bool(np.allclose(w,data['weights'][i,arm],atol=1e-5,rtol=1e-6))
            mu_ok &= bool(np.allclose(mu,data['mu'][i,arm],atol=1e-5,rtol=1e-6))
            std_ok &= bool(np.allclose(std,data['std'][i,arm],atol=1e-5,rtol=1e-6))
            masses_ok &= bool(np.allclose(masses[i,arm],data['masses'][i,arm],atol=1e-5,rtol=1e-6))
            counts_ok &= bool(counts[i,arm] == data['counts'][i,arm])
    checks.update({'elite_top64_valid':top_set_ok,'weights_replay':weights_ok,
                   'mu_replay':mu_ok,'std_replay':std_ok,'masses_replay':masses_ok,'counts_replay':counts_ok})
    j = data['mu_values'].astype(np.float64)
    gfp = j[:,0]-j[:,2]
    gq = j[:,1]-j[:,2]
    binding = (gfp>1e-4)&(masses[:,0]>masses[:,2]+1e-4)&(~cutoff_ties.any(axis=1))
    joint = binding&(gq<=.9*gfp)&(masses[:,1]<=.9*masses[:,0])
    science = all(checks.values())
    decision = ('implementation_inconclusive' if not science else
                'inconclusive_binding' if int(binding.sum())<4 else
                'scope_limited_preliminary_go' if int(joint.sum())>=4 else 'mechanism_no_go')
    return {'decision':decision,'checks':checks,'binding_count':int(binding.sum()),
            'joint_count':int(joint.sum()),'common_scores_max_abs_difference':float(np.max(np.abs(values[:,1:,24:]-values[:,:1,24:]))),
            'states':[{'reset_seed':5217+i,'J_by_arm':j[i].tolist(),'G_FP':float(gfp[i]),
                       'G_W4':float(gq[i]),'delta_support':float(gq[i]-gfp[i]),
                       'mass_by_arm':masses[i].tolist(),'elite_count_by_arm':counts[i].tolist(),
                       'cutoff_ties_by_arm':cutoff_ties[i].tolist(),'binding':bool(binding[i]),'joint_damage':bool(joint[i]),
                       'policy_action_mse_FP_W4':float(np.square(actions[i,1,:,:24]-actions[i,0,:,:24]).mean()),
                       'saturated_fraction_by_arm':(np.abs(actions[i,:,:,:24])>=1-1e-6).mean(axis=(1,2,3)).tolist()}
                      for i in range(8)],
            'boundary':'First internal update under learned FP score; not return or complete planner validation.'}
