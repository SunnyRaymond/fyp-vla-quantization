"""Official AQLM initialization and matched frozen-code loss arms; no environments."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import json
import math
import os
from pathlib import Path
import random
import signal
import time

PROTOCOL = 'fastwam-aqlm-loss-pair-v1'
ARMS = ('reconstruction', 'ours')
CASES = list(range(2000, 2080))
HELDOUT = list(range(3000, 3040))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def validate(p):
    t, q = p['training'], p['quantization']
    if (p['protocol'] != PROTOCOL or p['arms'] != list(ARMS)
            or p['calibration']['case_ids'] != CASES or p['target_linears'] != 614
            or p['heldout']['case_ids'] != HELDOUT or p['heldout']['seed_indices'] != [0, 1]
            or t['updates'] != 8 or t['contexts_per_update'] != 10
            or q['in_group_size'] != 8 or q['num_codebooks'] != 4
            or q['nbits_per_codebook'] != 8 or q['native_kernel']
            or p['boundary']['closed_loop_authorized'] or p['expected']['closed_loop_episodes'] != 0):
        raise ValueError('Frozen AQLM paired protocol differs')
    budgets={
        'light_v2':dict(init_max_iter=20,init_max_points_per_centroid=64,max_cluster_vectors=16384,
            max_epochs=10,steps_per_epoch=20,relative_mse_tolerance=.005,beam_size=1),
        'minimal_v3':dict(init_max_iter=5,init_max_points_per_centroid=16,max_cluster_vectors=4096,
            max_epochs=1,steps_per_epoch=5,relative_mse_tolerance=.005,beam_size=1),
    }
    expected=budgets.get(p['fitting'].get('profile'))
    if expected and any(p['fitting'].get(k) != v for k,v in expected.items()):
        raise ValueError('User-requested AQLM fit budget differs')


def fit_settings(p):
    return {k:p['fitting'].get(k) for k in ('init_max_iter','init_max_points_per_centroid',
        'max_cluster_vectors','max_epochs','steps_per_epoch','relative_mse_tolerance','beam_size','lr')}


def beta_from_means(p, latent, kl):
    if any(not math.isfinite(v) or v <= 0 for v in (latent, kl)):
        raise ValueError('Initial calibration loss must be finite and positive')
    return p['reader_source']['config']['coefficient_fraction'] * p['training']['latent_weight'] * latent / kl


def load_capture_case(torch, captures, case, names):
    receipt = read(captures/'cases'/f'{case}.json')
    if (receipt['protocol'] != PROTOCOL or receipt['case_id'] != case
            or set(receipt['calls']) != set(names)):
        raise ValueError('Activation capture identity or coverage differs')
    if receipt.get('layout') == 'context_archive_v2':
        payload = torch.load(captures/'archives'/f'{case}.pt',map_location='cpu',weights_only=True)
        if payload['protocol'] != PROTOCOL or payload['case_id'] != case or set(payload['rows']) != set(names):
            raise ValueError('Activation archive identity or coverage differs')
        return payload['rows']
    return {name:torch.load(captures/f'{i:04d}'/f'{case}.pt',map_location='cpu',weights_only=True)
            for i,name in enumerate(names)}


def pack_capture_rows(torch, rows, names):
    joined={name:torch.cat(rows[name]) for name in names}
    flat=torch.cat([joined[name].reshape(-1) for name in names]).cpu()
    result,offset={},0
    for name in names:
        value=joined[name]
        result[name]=flat[offset:offset+value.numel()].reshape(value.shape)
        offset+=value.numel()
    return result


def capture_packing_selfcheck(torch):
    rows={'a':[torch.arange(6,device='cuda',dtype=torch.bfloat16).reshape(2,3),
               torch.ones(1,3,device='cuda',dtype=torch.bfloat16)],
          'b':[torch.arange(4,device='cuda',dtype=torch.bfloat16).reshape(1,4)]}
    packed=pack_capture_rows(torch,rows,['a','b'])
    if not all(packed[n].dtype == rows[n][0].dtype and torch.equal(packed[n],torch.cat(rows[n]).cpu()) for n in rows):
        raise ValueError('Batched activation transfer changed sampled rows')
    return {'passed':True,'bitwise_equal':True}


def official_quantize(torch, engine, arguments):
    # Stock AQLM K-means uses CUDA index_reduce; the one common fit is shared by both arms.
    enabled=torch.are_deterministic_algorithms_enabled()
    warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        return engine.quantize(args=arguments,verbose=False)
    finally:
        torch.use_deterministic_algorithms(enabled,warn_only=warn_only)


class Work:
    def __init__(self, p, stage, arm=None):
        import sampler
        sampler._require_compute_allocation()
        validate(p)
        import torch
        import numpy as np
        import runner
        import calibration_source
        import source_worker
        import latent_reference
        from checkpointing import atomic_torch_save, capture_rng, restore_rng, to_cpu, optimizer_to_device
        self.p, self.stage, self.arm = p, stage, arm
        self.torch, self.np, self.runner, self.source = torch, np, runner, calibration_source
        self.sampler, self.helpers, self.latent_reference = sampler, source_worker, latent_reference
        self.atomic, self.capture_rng, self.restore_rng = atomic_torch_save, capture_rng, restore_rng
        self.to_cpu, self.optimizer_to_device = to_cpu, optimizer_to_device
        self.root = Path(p['remote_root'])
        self.out = self.root/'results'
        self.unit = self.out/('arms/'+arm if arm else stage)
        self.unit.mkdir(parents=True, exist_ok=True)
        self.job = os.environ['PBS_JOBID']
        self.started = time.monotonic()
        self.deadline = float(os.environ['FW_JOB_START_EPOCH']) + 24*3600 - 900
        self.stopped = []
        signal.signal(signal.SIGTERM, lambda *args: self.stopped.append(True))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        self.reset_rng()
        plan = read(p['calibration']['plan_file'])
        self.rows = {r['case_id']: r for r in plan['inputs']}
        if plan['calibration_cohorts']['c80'] != CASES:
            raise ValueError('Original calibration contexts differ')
        self.schedule = source_worker.validate_schedule(
            [r['cases'] for r in read(p['training']['schedule_source'])['arms'][p['source_rotation']['arm']]['history']],
            self.rows, p['suites'])
        self.teacher_root = Path(p['c80_teacher_directory'])
        self.heldout_teacher_root = Path(p['baseline_root'])/'results/teacher'
        for directory, cases, seeds in ((self.teacher_root, CASES, [0]), (self.heldout_teacher_root, HELDOUT, [0, 1])):
            receipt = read(directory/'receipt.json')
            if not receipt.get('complete') or receipt['case_ids'] != cases or receipt['seed_indices'] != seeds:
                raise ValueError('Existing teacher cache differs')
        self.data = {}
        self.counters = dict(student_queries=0, reference_queries=0, reader_queries=0, parameter_reverse_passes=0)
        if (self.unit/'query_counters.json').exists():
            self.counters.update(read(self.unit/'query_counters.json')['counters'])
        from aqlm_banks import selfcheck
        self.decoder_checks = selfcheck()
        save(self.unit/'decoder_selfcheck.json', self.decoder_checks)
        if stage == 'prepare':
            save(self.unit/'capture_packing_selfcheck.json',capture_packing_selfcheck(torch))
        self.progress('loading')
        self.runtime = runner.Runtime(self.unit)
        self.runtime.arm('bf16')
        for v in self.runtime.model.parameters():
            v.requires_grad_(False)
        from aqlm_banks import AQLMBanks
        self.bank = AQLMBanks(self.runtime, p, self.out/'banks', load_quantized=stage != 'prepare')
        self.masters = self.bank.masters
        self.reset_rng()

    def reset_rng(self):
        seed = self.p['training']['random_seed']
        random.seed(seed)
        self.np.random.seed(seed)
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)

    def enough(self, reserve=300):
        if self.stopped or time.time()+reserve >= self.deadline:
            raise TimeoutError('Committed AQLM units retained for resume')

    def progress(self, phase, **values):
        record = dict(protocol=PROTOCOL, stage=self.stage, arm=self.arm, pbs_jobid=self.job,
            phase=phase, seconds=time.monotonic()-self.started, counters=self.counters, **values)
        save(self.unit/'progress.json', record)
        print('AQLM_PROGRESS '+json.dumps(record, allow_nan=False), flush=True)

    def count(self, name):
        self.counters[name] += 1
        save(self.unit/'query_counters.json', dict(protocol=PROTOCOL, counters=self.counters))

    def datum(self, case):
        return self.helpers.Work.datum(self, case)

    def teacher(self, case, seed=0):
        return self.helpers.Work.teacher(self, case, seed)

    def loss(self, action, future, target):
        return self.helpers.Work.loss(self, action, future, target)

    def metrics(self, action, future, target):
        return self.helpers.Work.metrics(self, action, future, target)

    def query(self, case, seed=0, grad=False):
        self.enough()
        value, description = self.datum(case)
        self.runtime.description = description
        self.bank.seen.clear()
        self.bank.rotations.refresh()
        self.count('reference_queries' if self.bank.mode == 'bf16' else 'student_queries')
        with self.torch.set_grad_enabled(grad):
            action, counts, future = self.sampler.differentiable_action(
                self.runtime, value, self.rows[case]['sampler_seeds'][seed], checkpointed=grad, return_future=True)
        if len(self.bank.seen) != 614 or counts != dict(video_steps=10, action_steps=10):
            raise ValueError('Incomplete target or sampler coverage')
        return action, future, counts

    def prepare(self):
        torch = self.torch
        captures = self.out/'captures'
        names = list(self.bank.modules)
        # A resumed capture must reproduce the frozen teacher after storage-only fixes.
        if (captures/'cases'/f'{CASES[0]}.json').exists():
            action,future,_=self.query(CASES[0])
            target=self.teacher(CASES[0])
            error=max(float((action.float().cpu()-target['action'].float()).abs().max()),
                      float((future.float().cpu()-target['future'].float()).abs().max()))
            if error > 1e-6:
                raise ValueError('Resumed BF16 capture changed teacher output')
            save(self.unit/'resume_identity.json',dict(protocol=PROTOCOL,pbs_jobid=self.job,max_abs=error))
            del action,future,target
        for case in CASES:
            receipt = captures/'cases'/f'{case}.json'
            if receipt.exists():
                row = read(receipt)
                if row['protocol'] != PROTOCOL or len(row['calls']) != 614:
                    raise ValueError('Incomplete activation capture')
                continue
            self.enough()
            case_started=time.monotonic()
            rows, calls, hooks = {}, Counter(), []
            def capture(name):
                def hook(module, inputs):
                    x = self.bank.operand(inputs[0], name).detach().reshape(-1, module.in_features)
                    ids = torch.linspace(0, x.shape[0]-1, min(self.p['fitting']['rows_per_linear_call'], x.shape[0]),
                                         device=x.device).long()
                    sampled = x[ids]
                    padded = (module.in_features+7)//8*8
                    sampled = torch.nn.functional.pad(sampled, (0, padded-module.in_features))
                    rows.setdefault(name, []).append(sampled.detach())
                    calls[name] += 1
                return hook
            try:
                for name, module in self.bank.modules.items():
                    hooks.append(module.register_forward_pre_hook(capture(name)))
                self.bank.mode = 'bf16'
                action, future, _ = self.query(case)
                forward_seconds=time.monotonic()-case_started
            finally:
                for hook in hooks:
                    hook.remove()
            if set(rows) != set(names):
                raise ValueError('Capture missed a target Linear')
            identity = None
            if case == CASES[0]:
                target = self.teacher(case)
                identity = max(float((action.float().cpu()-target['action'].float()).abs().max()),
                    float((future.float().cpu()-target['future'].float()).abs().max()))
                if identity > 1e-6:
                    raise ValueError('BF16 capture differs from frozen teacher')
            directory=captures/'archives'
            directory.mkdir(parents=True,exist_ok=True)
            copy_started=time.monotonic()
            packed=pack_capture_rows(torch,rows,names)
            copy_seconds=time.monotonic()-copy_started
            del rows
            tensor_bytes=sum(v.numel()*v.element_size() for v in packed.values())
            save_started=time.monotonic()
            self.atomic(torch,directory/f'{case}.pt',dict(protocol=PROTOCOL,case_id=case,rows=packed))
            save_seconds=time.monotonic()-save_started
            save(receipt, dict(protocol=PROTOCOL,case_id=case,calls=dict(calls),identity_max_abs=identity,
                layout='context_archive_v2',tensor_bytes=tensor_bytes,forward_seconds=forward_seconds,save_seconds=save_seconds))
            self.progress('capture',contexts=case-CASES[0]+1,targets=614,forward_seconds=forward_seconds,
                copy_seconds=copy_seconds,save_seconds=save_seconds,tensor_bytes=tensor_bytes)
            del packed
            del action, future
        from aq_engine import AQEngine
        from argparse import Namespace
        arguments = {k:self.p['fitting'][k] for k in ('init_max_iter','init_max_points_per_centroid','lr',
            'max_epochs','steps_per_epoch','relative_mse_tolerance','beam_size','print_frequency')}
        arguments.update({k:self.p['quantization'][k] for k in ('in_group_size','out_group_size','num_codebooks',
            'nbits_per_codebook','codebook_value_nbits','codebook_value_num_groups','scale_nbits')})
        arguments['devices'] = [torch.device('cuda:0')]
        manifest = []
        bank_root = self.out/'banks'
        bank_root.mkdir(exist_ok=True)
        capture_cache={}
        if any(not (bank_root/f'{i:04d}.pt').exists() for i in range(len(names))):
            host_bytes=0
            limit=int(self.p['execution']['prepare']['memory_gb']*1e9*.55)
            for case in CASES:
                self.enough()
                capture_cache[case]=load_capture_case(torch,captures,case,names)
                host_bytes+=sum(v.numel()*v.element_size() for v in capture_cache[case].values())
                if host_bytes > limit:
                    raise RuntimeError('Full C80 activation cache exceeds the declared host-memory reserve')
                self.progress('activation_cache',contexts=len(capture_cache),host_tensor_bytes=host_bytes)
        for index, (name, module) in enumerate(self.bank.modules.items()):
            path = bank_root/f'{index:04d}.pt'
            if path.exists():
                item = torch.load(path, map_location='cpu', weights_only=False)
                if item['module'] != name or item['protocol'] != PROTOCOL or item['official_commit'] != self.p['official']['commit']:
                    raise ValueError('Shared official bank identity differs')
            else:
                self.enough(reserve=1800)
                self.reset_rng()
                reference = self.bank.reference_weight(name).detach().float()
                if reference.numel()//8 < 256:
                    raise ValueError('Tiny layer requires explicitly reviewed K256 initialization: '+name)
                layer = torch.nn.Linear(reference.shape[1], reference.shape[0], bias=False, device='meta')
                layer.weight = torch.nn.Parameter(reference, requires_grad=False)
                engine = AQEngine(layer)
                for case in CASES:
                    x = capture_cache[case][name].to('cuda')
                    engine.add_batch(x)
                self.progress('fitting',fitted=index,total=614,module=name,
                    common_fit_deterministic_algorithms=False,shared_initialization=True)
                quantized = official_quantize(torch,engine,Namespace(**arguments))
                fit_mse = float(engine._compute_mse().detach())
                quantized = quantized.cpu()
                quantized.codes = torch.nn.Parameter(quantized.get_codes().to(torch.uint8), requires_grad=False)
                quantized.codebooks.data = quantized.codebooks.detach().to(torch.bfloat16)
                quantized.scales.data = quantized.scales.detach().to(torch.bfloat16)
                original = list(module.weight.shape)
                bits = quantized.codes.numel()*8 + quantized.codebooks.numel()*16 + quantized.scales.numel()*16
                preprocessing_bits = sum(v.numel()*v.element_size()*8 for v in self.bank.transforms[name].values()
                    if isinstance(v, torch.Tensor))
                item = dict(protocol=PROTOCOL, module=name, module_index=index, original_shape=original,
                    padded_shape=[reference.shape[0], reference.shape[1]], official_commit=self.p['official']['commit'],
                    quantized_weight=quantized, fit_mse=fit_mse, calibration_rows=engine.nsamples,
                    encoded_bits=bits, preprocessing_bits=preprocessing_bits,
                    original_parameters=math.prod(original), common_fit_deterministic_algorithms=False,
                    fit_profile=self.p['fitting'].get('profile','heavy_v1'),fit_settings=fit_settings(self.p),
                    pbs_jobid=self.job)
                self.atomic(torch, path, item)
                del quantized, engine, layer, reference, x
            record={k:v for k,v in item.items() if k != 'quantized_weight'}
            record.setdefault('fit_profile','heavy_v1_retained')
            manifest.append(record)
            self.progress('fitting', fitted=index+1, total=614)
            del item
            gc.collect()
            torch.cuda.empty_cache()
        save(self.out/'banks/manifest.json', dict(protocol=PROTOCOL, targets=manifest))
        rotation_bits = sum(v.numel()*v.element_size()*8 for v in self.bank.rotations.snapshot().values())
        parameters = sum(r['original_parameters'] for r in manifest)
        save(self.unit/'summary.json', dict(protocol=PROTOCOL, complete=True, pbs_jobid=self.job,
            target_linears=614, fit_banks=614, calibration_contexts=80, official_commit=self.p['official']['commit'],
            weight_encoding_bpw=sum(r['encoded_bits'] for r in manifest)/parameters,
            effective_bpw=(sum(r['encoded_bits']+r['preprocessing_bits'] for r in manifest)+rotation_bits)/parameters,
            rotation_bits=rotation_bits,
            fit_profile_counts=dict(Counter(r['fit_profile'] for r in manifest)),
            requested_fit_settings=fit_settings(self.p),
            counters=self.counters, closed_loop_episodes=0))

    def reader_action(self, reader, case, future, grad, seed=0):
        self.enough()
        self.count('reader_queries')
        value, description = self.datum(case)
        self.runtime.description = description
        with self.torch.set_grad_enabled(grad):
            return reader(self.runtime, value, self.rows[case]['sampler_seeds'][seed], future, checkpointed=grad)

    def reader_preparation(self):
        from reader_runtime import reader_session
        from reader_loss import reader_terms
        torch, p = self.torch, self.p
        directory = self.out/'reader_preparation'
        if (directory/'summary.json').exists():
            saved = read(directory/'summary.json')
            if saved['protocol'] != p or saved['cases'] != CASES:
                raise ValueError('Reader coefficient recipe differs')
            return saved
        stats = {'action':p['reader_source']['action_stats']}
        for case in CASES:
            path = directory/'inputs'/f'{case}.pt'
            if path.exists():
                continue
            action, future, _ = self.query(case)
            target = self.teacher(case)
            terms = self.metrics(action, future, target)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.atomic(torch, path, dict(future=future.detach().cpu(), metrics=terms))
            self.progress('initial_student', contexts=case-CASES[0]+1)
            del action, future, target
        records = []
        with reader_session(torch, self.runtime, self.bank, self.masters) as reader:
            for case in CASES:
                path = directory/'cases'/f'{case}.json'
                if path.exists():
                    records.append(read(path))
                    continue
                saved = torch.load(directory/'inputs'/f'{case}.pt', map_location='cpu', weights_only=True)
                target = self.teacher(case)
                if case == CASES[0]:
                    identity, _ = self.reader_action(reader, case, target['future'].cuda(), False)
                    error = float((identity.float().cpu()-target['action'].float()).abs().max())
                    if error > 1e-6:
                        raise ValueError('Frozen reader teacher identity failed')
                leaf = saved['future'].cuda().float().detach().requires_grad_(True)
                action, _ = self.reader_action(reader, case, leaf, True)
                terms = reader_terms(torch, action, target['action'], stats, p['reader_source']['config'])
                gradient, = torch.autograd.grad(terms['reader_kl'], leaf)
                if not torch.isfinite(gradient).all() or not bool((gradient[:,:,1:] != 0).any()):
                    raise ValueError('Reader gradient is disconnected/nonfinite')
                row = dict(case_id=case, latent_nmse=saved['metrics']['future_latent_nmse'],
                    reader_kl=float(terms['reader_kl'].detach()), gradient_norm=float(gradient.norm()))
                save(path, row)
                records.append(row)
                self.progress('initial_reader', contexts=len(records))
                del leaf, action, terms, gradient, target
        self.bank.bind_dense()
        beta = beta_from_means(p, sum(r['latent_nmse'] for r in records)/80,
            sum(r['reader_kl'] for r in records)/80)
        saved = dict(protocol=p, complete=True, cases=CASES, stats=stats, beta=beta, heldout_used=False)
        save(directory/'summary.json', saved)
        return saved

    def ours_batch(self, batch, beta, stats):
        from reader_runtime import reader_session
        from reader_loss import reader_terms
        torch, staged, metrics = self.torch, [], []
        for case in batch:
            action, future, _ = self.query(case)
            target = self.teacher(case)
            a = action.detach().float().requires_grad_(True)
            z = future.detach().float().requires_grad_(True)
            total, _, _ = self.loss(a, z, target)
            ga, gz = torch.autograd.grad(total, (a,z))
            metrics.append(self.metrics(action, future, target))
            staged.append(dict(case=case, action=action.detach().cpu(), future=future.detach().cpu(),
                ga=ga.detach().cpu(), gz=gz.detach().cpu()))
            del action, future, total, a, z, ga, gz
        reader_values = []
        with reader_session(torch, self.runtime, self.bank, self.masters) as reader:
            for row in staged:
                leaf = row['future'].cuda().float().detach().requires_grad_(True)
                action, _ = self.reader_action(reader, row['case'], leaf, True)
                terms = reader_terms(torch, action, self.teacher(row['case'])['action'], stats, self.p['reader_source']['config'])
                rg, = torch.autograd.grad(beta*terms['reader_kl'], leaf)
                if not torch.isfinite(rg).all() or bool((rg[:,:,:1] != 0).any()) or not bool((rg[:,:,1:] != 0).any()):
                    raise ValueError('Invalid reader future cotangent')
                row['gz'] += rg.detach().cpu()
                reader_values.append(float(terms['reader_kl'].detach()))
                del leaf, action, terms, rg
        self.bank.bind_dense()
        error = 0.
        for row in staged:
            action, future, _ = self.query(row['case'], grad=True)
            delta = max(float((action.detach().float().cpu()-row['action'].float()).abs().max()),
                float((future.detach().float().cpu()-row['future'].float()).abs().max()))
            if delta > 1e-6:
                raise ValueError('Staged student replay differs')
            error = max(error, delta)
            torch.autograd.backward((action,future), ((row['ga']/10).to(action),(row['gz']/10).to(future)))
            self.count('parameter_reverse_passes')
            del action, future
        return dict(before={k:sum(r[k] for r in metrics)/10 for k in metrics[0]},
            reader_kl=sum(reader_values)/10, staged_replay_max_abs=error)

    def reconstruction_batch(self, batch):
        """AQEngine's original Linear-output objective, streamed on fixed teacher inputs."""
        torch, losses = self.torch, []
        self.bank.mode = 'bf16'
        try:
            for case in batch:
                expected = read(self.out/'captures/cases'/f'{case}.json')['calls']
                observed, hooks = Counter(), []
                def reconstruction(name):
                    def hook(module, inputs, output):
                        observed[name] += 1
                        with torch.inference_mode(False), torch.enable_grad():
                            x = inputs[0].detach().clone()
                            qx = self.bank.operand(x, name).detach()
                            with torch.no_grad():
                                reference = self.bank.reference_weight(name, cache_bf16=True)[:,:module.in_features]
                                target = torch.nn.functional.linear(qx, reference, module.bias)
                            estimate = torch.nn.functional.linear(qx, self.bank.weight(name), module.bias)
                            loss = (estimate.float()-target.float()).square().mean()
                            if not torch.isfinite(loss):
                                raise ValueError('Nonfinite layer reconstruction')
                            (loss/(614*10*expected[name])).backward()
                            losses.append(float(loss.detach())/expected[name]/614/10)
                    return hook
                try:
                    for name, module in self.bank.modules.items():
                        hooks.append(module.register_forward_hook(reconstruction(name)))
                    self.query(case)
                finally:
                    for hook in hooks:
                        hook.remove()
                if dict(observed) != expected:
                    raise ValueError('Reference layer-call weighting differs')
                self.count('parameter_reverse_passes')
        finally:
            self.bank.mode = 'aqlm'
        return dict(layer_output_mse=sum(losses))

    def train(self):
        torch, p = self.torch, self.p
        checks = self.decoder_checks
        if not checks['passed']:
            raise ValueError('Official decoder/custom backward parity failed')
        self.bank.mode = 'aqlm'
        action, future, _ = self.query(CASES[0])
        self.atomic(torch, self.unit/'initial_anchor.pt', dict(action=action.detach().cpu(), future=future.detach().cpu()))
        save(self.unit/'setup_checks.json', dict(protocol=PROTOCOL, arm=self.arm, passed=True,
            target_linears=614, decoder=checks, initial_metrics=self.metrics(action,future,self.teacher(CASES[0]))))
        del action, future
        prep = self.reader_preparation() if self.arm == 'ours' else dict(beta=0,stats={})
        beta = prep['beta']
        optimizer = torch.optim.Adam(self.bank.parameters(), lr=p['training']['learning_rate'], betas=tuple(p['training']['betas']))
        state = dict(updates=0, exposures=0, history=[], lineage=[])
        checkpoint = self.unit/'checkpoint.pt'
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if saved['protocol'] != p or saved['arm'] != self.arm or saved['beta'] != beta:
                raise ValueError('Committed arm protocol differs')
            self.bank.restore(saved['parameters'])
            state = saved['state']
            optimizer.load_state_dict(saved['optimizer'])
            self.optimizer_to_device(torch, optimizer, 'cuda')
            self.restore_rng(torch, saved['rng'])
        state['lineage'] = list(dict.fromkeys(state['lineage']+[self.job]))
        while state['updates'] < 8:
            self.enough(reserve=1200)
            update, started = state['updates'], time.monotonic()
            batch = self.schedule[update]
            optimizer.zero_grad(set_to_none=True)
            self.progress('training', updates=update, exposures=state['exposures'])
            values = self.reconstruction_batch(batch) if self.arm == 'reconstruction' else self.ours_batch(batch,beta,prep['stats'])
            coverage = self.bank.gradient_report()
            if coverage['connected'] != 614 or coverage['finite'] != 614 or coverage['nonzero'] < 1:
                raise ValueError('Codebook/scale gradient coverage failed')
            if any(v.grad is not None for v in self.runtime.model.parameters()) or any(v.grad is not None for v in self.bank.rotations.parameters()):
                raise ValueError('Frozen base or rotation accumulated a gradient')
            norm = torch.nn.utils.clip_grad_norm_(self.bank.parameters(), p['training']['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            if update == 0 and p['training']['optimizer_reset_after_update1']:
                optimizer.state.clear()
            self.bank.refresh()
            state['updates'] += 1
            state['exposures'] += 10
            state['history'].append(dict(update=state['updates'], cases=batch, **values, coverage=coverage,
                gradient_norm=float(norm), seconds=time.monotonic()-started, pbs_jobid=self.job))
            self.atomic(torch, checkpoint, dict(protocol=p,arm=self.arm,beta=beta,state=state,
                parameters=self.bank.snapshot(),optimizer=self.to_cpu(optimizer.state_dict()),rng=self.capture_rng(torch)))
            save(self.unit/'training.json',dict(protocol=PROTOCOL,arm=self.arm,**state))
            self.progress('training_committed',updates=state['updates'],exposures=state['exposures'])
            gc.collect()
            torch.cuda.empty_cache()
        for v in self.bank.parameters():
            v.requires_grad_(False)
            v.grad = None
        optimizer.state.clear()
        self.bank.release_reference_cache()
        packed = {name:{key:value.to(torch.bfloat16) for key,value in row.items()}
                  for name,row in self.bank.snapshot().items()}
        self.atomic(torch,self.unit/'deployment.pt',dict(protocol=p,arm=self.arm,updates=8,exposures=80,
            parameters=packed,banks=str(self.out/'banks'),beta=beta))
        # Stored BF16 values must reproduce the forward before evaluating them.
        self.bank.restore(packed)
        for split, cases, seeds in (('calibration',CASES,[0]),('heldout',HELDOUT,[0,1])):
            for case in cases:
                for seed in seeds:
                    path=self.unit/'evaluation'/split/f'{case}_{seed}.json'
                    if path.exists():
                        saved=read(path)
                        if saved['protocol'] != PROTOCOL or saved['updates'] != 8 or saved['arm'] != self.arm:
                            raise ValueError('Existing evaluation unit differs')
                        continue
                    action,future,_=self.query(case,seed)
                    values=self.metrics(action,future,self.teacher(case,seed))
                    save(path,dict(protocol=PROTOCOL,arm=self.arm,case_id=case,seed_index=seed,updates=8,
                        suite=self.rows[case]['suite'],metrics=values,pbs_jobid=self.job))
                    self.progress('evaluation',split=split,case=case,seed=seed,updates=8)
                    del action,future
        save(self.unit/'summary.json',dict(protocol=PROTOCOL,arm=self.arm,complete=True,updates=8,exposures=80,
            target_linears=614,calibration_queries=80,heldout_queries=80,beta=beta,pbs_jobid=self.job,
            lineage=state['lineage'],counters=self.counters,closed_loop_episodes=0))
        (self.unit/'failure.json').unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--stage',required=True,choices=('prepare','arm'))
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args()
    p=read(args.protocol)
    work=None
    try:
        work=Work(p,args.stage,args.arm)
        work.prepare() if args.stage=='prepare' else work.train()
    except TimeoutError as error:
        if work is not None:
            work.progress('checkpointed',reason=str(error))
        raise SystemExit(75)
    except Exception as error:
        import traceback
        directory=Path(p['remote_root'])/'results'/('arms/'+args.arm if args.arm else args.stage)
        save(directory/'failure.json',dict(protocol=PROTOCOL,pbs_jobid=os.environ.get('PBS_JOBID'),
            error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
