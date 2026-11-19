"""Exact analytic reverse sampler for the declared frozen finite room reference.

No neural residual is accepted. Exactness refers to the known reference and
exact terminal initialization, subject to floating-point numerical error. The
Markov curve is defined by the immutable reference abar, not a silently changed
neural sampler. A fresh posterior room latent is shared at each reverse step.
"""
import torch
from privacy_synth.spatial_room_composition_model import FrozenRoomCompositionReference
from privacy_synth.spatial_joint_composition_reference import composition_posterior
from privacy_synth.spatial_set_diffusion import absorbing_reverse


class ReferenceMarkovCurve:
    def __init__(self,point):
        bar=point.bar
        if bar.shape!=(65,) or float(bar[0])!=1 or not bool(torch.isfinite(bar).all()) or not bool(((bar[1:]>0)&(bar[1:]<bar[:-1])).all()):raise ValueError('Original positive decreasing64 reference abar required')
        self.bar=bar
        # Ratios in double avoid subtractive loss before the final dtype cast.
        self.alpha=torch.cat([torch.ones_like(bar[:1]),(bar[1:].double()/bar[:-1].double()).to(bar.dtype)])
        self.beta=torch.cat([torch.zeros_like(bar[:1]),(1-bar[1:].double()/bar[:-1].double()).to(bar.dtype)])
        self.steps=64


def _point(reference):
    if not isinstance(reference,FrozenRoomCompositionReference) or reference.training or any(p.requires_grad for p in reference.parameters()):raise ValueError('Frozen eval room-composition reference only; no neural residual model')
    return reference.joint.point


def _categorical(probability,uniform):
    if probability.shape[:-1]!=uniform.shape or not bool(torch.isfinite(probability).all()) or not bool(torch.isfinite(uniform).all()) or bool((probability<0).any()) or bool(((uniform<0)|(uniform>=1)).any()):raise ValueError('Finite matched probabilities/uniform[0,1) required')
    cdf=probability.cumsum(-1)
    if bool((cdf[...,-1]<=0).any()):raise ValueError('No zero-mass branch or fallback')
    cdf=cdf/cdf[...,-1:]
    return (cdf<=uniform[...,None]).sum(-1)


def _rotate(vectors,values):
    return torch.einsum('qe,qde->qd',values,vectors)


@torch.no_grad()
def clean_draw(reference,condition,count,h_uniform,class_uniform,component_uniform,normal,*,return_details=False):
    point=_point(reference);B=len(condition)
    if not isinstance(count,int) or isinstance(count,bool) or count<1 or h_uniform.shape!=(B,) or class_uniform.shape!=(B,count) or component_uniform.shape!=(B,count) or normal.shape!=(B,count,3):raise ValueError('Explicit positive count and matched clean-draw arrays required')
    if normal.dtype!=point.centers.dtype or normal.device!=point.centers.device or not bool(torch.isfinite(normal).all()):raise ValueError('Finite reference dtype/device Gaussian noise required')
    logits=reference.mixing_logits(condition);H=logits.shape[1];h=_categorical(torch.full((B,H),1/H,device=normal.device,dtype=normal.dtype),h_uniform);pi=logits[torch.arange(B,device=normal.device),h].softmax(-1)
    labels=_categorical(pi[:,None,:].expand(B,count,-1),class_uniform);flat=labels.reshape(-1);u=component_uniform.reshape(-1);eps=normal.reshape(-1,3);out=[];chosen=[]
    for start in range(0,len(flat),point.query_chunk):
        end=start+point.query_chunk;allowed=point.component_class[None]==flat[start:end,None];weights=torch.where(allowed,point.log_component_prior[None],-torch.inf).softmax(-1);j=_categorical(weights,u[start:end]);U=point.eigenvectors[j];projected=torch.einsum('qd,qde->qe',eps[start:end],U);z=point.centers[j]+_rotate(U,projected*point.eigenvalues[j].sqrt());out.append(z);chosen.append(j)
    xyz=torch.cat(out).reshape(B,count,3)
    return (xyz,labels,dict(h=h,component=torch.cat(chosen).reshape(B,count))) if return_details else (xyz,labels)


@torch.no_grad()
def sample_clean(reference,condition,count,generator):
    B=len(condition);kw=dict(device=condition.device,dtype=condition.dtype)
    return clean_draw(reference,condition,count,torch.rand((B,),generator=generator,**kw),torch.rand((B,count),generator=generator,**kw),torch.rand((B,count),generator=generator,**kw),torch.randn((B,count,3),generator=generator,**kw))


@torch.no_grad()
def component_bridge_draw(point,noisy,clean_class,t,component_uniform,normal,*,return_components=False):
    if noisy.ndim!=3 or noisy.shape[-1]!=3 or clean_class.shape!=noisy.shape[:2] or component_uniform.shape!=clean_class.shape or normal.shape!=noisy.shape or not isinstance(t,int) or isinstance(t,bool) or not 1<=t<=64:raise ValueError('Matched3D component bridge inputs/time required')
    if noisy.dtype!=point.centers.dtype or noisy.device!=point.centers.device or normal.dtype!=noisy.dtype or normal.device!=noisy.device or clean_class.device!=noisy.device or bool(((clean_class<0)|(clean_class>=point.classes)).any()) or not bool(torch.isfinite(noisy).all() and torch.isfinite(normal).all()):raise ValueError('Finite registered reference queries/classes/noise required')
    curve=ReferenceMarkovCurve(point);bar=curve.bar[t];previous=curve.bar[t-1];alpha=curve.alpha[t];beta=curve.beta[t];flat=noisy.reshape(-1,3);labels=clean_class.reshape(-1);u=component_uniform.reshape(-1);eps=normal.reshape(-1,3);out=[];indices=[]
    vt=bar*point.eigenvalues+1-bar
    for start in range(0,len(flat),point.query_chunk):
        end=start+point.query_chunk;x=flat[start:end];delta=x[:,None]-bar.sqrt()*point.centers[None];projected=torch.einsum('qmd,mde->qme',delta,point.eigenvectors)
        logw=point.log_component_prior[None]-.5*((projected.square()/vt[None]).sum(-1)+vt.log().sum(-1)[None]);allowed=point.component_class[None]==labels[start:end,None];weights=torch.where(allowed,logw,-torch.inf).softmax(-1);j=_categorical(weights,u[start:end])
        U=point.eigenvectors[j];eig=point.eigenvalues[j];vp=previous*eig+1-previous;selected_vt=bar*eig+1-bar;gain=alpha.sqrt()*vp/selected_vt;variance=beta*vp/selected_vt
        centered=x-bar.sqrt()*point.centers[j];projection=torch.einsum('qd,qde->qe',centered,U);mean=previous.sqrt()*point.centers[j]+_rotate(U,projection*gain);projected_noise=torch.einsum('qd,qde->qe',eps[start:end],U)
        out.append(mean+_rotate(U,projected_noise*variance.sqrt()));indices.append(j)
    result=torch.cat(out).reshape_as(noisy)
    return (result,torch.cat(indices).reshape_as(clean_class)) if return_components else result


@torch.no_grad()
def reverse_step(reference,noisy,observed,condition,t,h_uniform,class_uniform,component_uniform,reveal_uniform,normal,*,return_details=False):
    point=_point(reference);B,N,_=noisy.shape
    if observed.shape!=(B,N) or h_uniform.shape!=(B,) or class_uniform.shape!=(B,N) or reveal_uniform.shape!=(B,N):raise ValueError('Matched room/object uniforms required')
    if bool(((reveal_uniform<0)|(reveal_uniform>=1)).any()) or not bool(torch.isfinite(reveal_uniform).all()):raise ValueError('Reveal uniform in[0,1) required')
    if not isinstance(t,int) or isinstance(t,bool) or not 1<=t<=64 or (t==64 and bool((observed<point.classes).any())):raise ValueError('Supported time; terminal must be allMASK')
    valid=torch.ones_like(observed,dtype=torch.bool);ts=torch.full((B,),t,device=noisy.device,dtype=torch.long);logL,mean_by_class=reference.joint.class_channels(noisy,ts);mix=reference.mixing_logits(condition);post=composition_posterior(logL,mean_by_class,mix,observed,valid)
    h=_categorical(post['log_posterior_h'].exp(),h_uniform);logits=mix[torch.arange(B,device=noisy.device),h][:,None,:]+logL;probability=logits.softmax(-1);known=observed<point.classes;hard=torch.nn.functional.one_hot(observed.clamp_max(point.classes-1),point.classes).to(noisy.dtype);probability=torch.where(known[:,:,None],hard,probability)
    proposed=_categorical(probability,class_uniform);xyz,j=component_bridge_draw(point,noisy,proposed,t,component_uniform,normal,return_components=True);labels=torch.where((~known)&(reveal_uniform<1/t),proposed,observed)
    return (xyz,labels,dict(h=h,proposed_class=proposed,component=j)) if return_details else (xyz,labels)


@torch.no_grad()
def initialize_forward(reference,condition,count,generator):
    """Only the exact noisy terminal array; original clean labels/latent not passed."""
    point=_point(reference);B=len(condition);kw=dict(device=condition.device,dtype=condition.dtype);clean,_=sample_clean(reference,condition,count,generator)
    noisy=point.bar[-1].sqrt()*clean+(1-point.bar[-1]).sqrt()*torch.randn((B,count,3),generator=generator,**kw)
    del clean
    return noisy


@torch.no_grad()
def sample_joint_reference(reference,condition,count,generator,check_stop=None):
    """Exact-reference terminal initialization, then64 shared-latent reverse steps."""
    point=_point(reference);B=len(condition);kw=dict(device=condition.device,dtype=condition.dtype);noisy=initialize_forward(reference,condition,count,generator)
    observed=torch.full((B,count),point.classes,device=condition.device,dtype=torch.long)
    for t in range(64,0,-1):
        if check_stop is not None:check_stop()
        noisy,observed=reverse_step(reference,noisy,observed,condition,t,torch.rand((B,),generator=generator,**kw),torch.rand((B,count),generator=generator,**kw),torch.rand((B,count),generator=generator,**kw),torch.rand((B,count),generator=generator,**kw),torch.randn((B,count,3),generator=generator,**kw))
    return noisy,observed


@torch.no_grad()
def sample_factorized_reference(reference,condition,count,generator,check_stop=None):
    """Kernel-only fixed-large control with SAME true terminal law/reference curve.

    This is not a reproduction of the old neural/FP32/N01-initialized pipeline.
    It isolates its factorized mean/Gaussian/category/final-step approximation
    while avoiding an initialization or reference-precision confound.
    """
    point=_point(reference);B=len(condition);kw=dict(device=condition.device,dtype=condition.dtype);curve=ReferenceMarkovCurve(point);noisy=initialize_forward(reference,condition,count,generator);observed=torch.full((B,count),point.classes,device=condition.device,dtype=torch.long);valid=torch.ones_like(observed,dtype=torch.bool)
    for t in range(64,0,-1):
        if check_stop is not None:check_stop()
        ts=torch.full((B,),t,device=condition.device,dtype=torch.long);velocity,logits=reference(noisy,observed,valid,condition,ts);normal=torch.randn((B,count,3),generator=generator,**kw)
        a=curve.bar[t];prev=curve.bar[t-1];clean=a.sqrt()*noisy-(1-a).sqrt()*velocity;mean=prev.sqrt()*curve.beta[t]/(1-a)*clean+curve.alpha[t].sqrt()*(1-prev)/(1-a)*noisy
        noisy=mean+curve.beta[t].sqrt()*normal if t>1 else mean
        observed=absorbing_reverse(observed,logits.softmax(-1),t,point.classes,generator)
    return noisy,observed
