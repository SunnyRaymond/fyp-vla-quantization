"""Match actual BF16-injected future NMSE without querying the action reader."""
import math


def nmse(torch,x,teacher):
    a=x[:,:,1:].double();b=teacher[:,:,1:].double()
    return float((a-b).square().sum()/b.square().sum().clamp_min(1e-12))


def match(torch,a,b,teacher,p,seed=0):
    if a.shape!=b.shape or a.shape!=teacher.shape or a.dtype!=torch.bfloat16 or b.dtype!=a.dtype or not teacher.is_floating_point():
        raise RuntimeError('Matching requires identical latent shapes and actual BF16-injected donors')
    if not all(bool(torch.isfinite(x).all()) for x in (a,b,teacher)):
        raise RuntimeError('Nonfinite source future')
    observed=teacher[:,:,:1].to(a.dtype)
    if not torch.equal(a[:,:,:1],observed) or not torch.equal(b[:,:,:1],observed):
        raise RuntimeError('Source observed frames differ')
    ea=nmse(torch,a,teacher);eb=nmse(torch,b,teacher)
    tol=max(p['matching_abs_tolerance'],p['matching_relative_tolerance']*eb)
    t=teacher[:,:,1:].double();e=a[:,:,1:].double()-t

    def construct(alpha):
        z=a.clone();z[:,:,1:]=(t+alpha*e).to(a.dtype)
        return z,nmse(torch,z,teacher)

    if ea==0:
        if eb!=0:raise RuntimeError('Cannot preserve zero A error direction with nonzero target')
        z,value=construct(0);alpha=0.;steps=0
    else:
        alpha=math.sqrt(eb/ea);z,value=construct(alpha);steps=0
        best=(abs(value-eb),alpha,z,value)
        low=0.;high=max(1.,2*alpha)
        while construct(high)[1]<eb:
            high*=2
            if not math.isfinite(high):raise RuntimeError('No finite matching bracket')
        for steps in range(p['matching_iterations']):
            if best[0]<=tol:break
            current=(low+high)/2;candidate,actual=construct(current)
            if abs(actual-eb)<best[0]:best=(abs(actual-eb),current,candidate,actual)
            if actual<eb:low=current
            else:high=current
        _,alpha,z,value=best
    rounding_before=value;edits=0
    if abs(value-eb)>tol:
        # A scalar ray crosses groups of BF16 rounding ties together. Break only
        # the rounding discretization, with bounded one-ULP edits and no reader.
        current=z[:,:,1:].contiguous().view(-1)
        target=t.contiguous().view(-1);direction=e.contiguous().view(-1)
        increase=value<eb
        toward=torch.where(direction>=0,torch.full_like(current,float('inf')),torch.full_like(current,-float('inf'))) if increase else target.to(current.dtype)
        adjacent=torch.nextafter(current,toward)
        den=float(t.square().sum().clamp_min(1e-12))
        changes=((adjacent.double()-target).square()-(current.double()-target).square())/den
        eligible=torch.nonzero(changes>0 if increase else changes<0).flatten()
        generator=torch.Generator().manual_seed(p['rounding_correction_seed']+int(seed))
        order=eligible[torch.randperm(len(eligible),generator=generator)]
        for index in order.tolist():
            change=float(changes[index]);new=value+change
            if abs(new-eb)>=abs(value-eb):continue
            current[index]=adjacent[index];value=new;edits+=1
            if abs(value-eb)<=tol or edits>=p['rounding_correction_max_coordinates']:break
        z[:,:,1:]=current.reshape(z[:,:,1:].shape);value=nmse(torch,z,teacher)
    error=z[:,:,1:].double()-t
    den=float(e.square().sum().sqrt()*error.square().sum().sqrt())
    cosine=float((e*error).sum())/den if den else None
    deviation=float((error-alpha*e).square().sum().sqrt())/max(float((alpha*e).square().sum().sqrt()),1e-12)
    check=dict(a_nmse=ea,b_nmse=eb,c_nmse=value,alpha=alpha,absolute_nmse_gap=abs(value-eb),
        relative_nmse_gap=abs(value-eb)/max(eb,1e-12),matching_tolerance=tol,direction_cosine=cosine,
        relative_bf16_direction_deviation=deviation,search_iterations=steps,
        rounding_correction_count=edits,rounding_correction_nmse_change=value-rounding_before,rounding_correction_seed=p['rounding_correction_seed']+int(seed),
        observed_frame_exact=bool(torch.equal(z[:,:,:1],observed)))
    if check['absolute_nmse_gap']>tol or not check['observed_frame_exact'] or (cosine is not None and cosine<p['direction_cosine_minimum']):
        raise RuntimeError(f'BF16 matching contract failed: {check}')
    return z,check


def gate(deltas):
    from statistics import mean,median
    if len(deltas)!=22:raise RuntimeError('Progression requires all22 paired test cases')
    values=list(deltas.values());loo=[mean(v for j,v in enumerate(values) if j!=i) for i in range(22)]
    return dict(mean_delta=mean(values),median_delta=median(values),improved=sum(v<0 for v in values),
        worsened=sum(v>0 for v in values),tied=sum(v==0 for v in values),
        leave_one_case_out_mean_min=min(loo),leave_one_case_out_mean_max=max(loo),
        passed=mean(values)<0 and median(values)<0 and sum(v<0 for v in values)>=12 and max(loo)<0)
