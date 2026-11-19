"""Borrow reference centered reverse innovations while preserving a frozen full mean.

This is a diagnostic proxy kernel, not the exact neural reverse distribution.
The original independent category update is retained, so even the zero-residual
multi-step *joint* reference law is not asserted. No new learned parameters.
"""
import torch
from privacy_synth.spatial_mixture_reference import GaussianMixtureReference
from privacy_synth.spatial_set_diffusion import absorbing_reverse


@torch.no_grad()
def reference_innovation(reference, noisy, observed, t, schedule, normal, component_uniform):
    """Zero-centered reference conditional mixture draw, including nonzero final variance.

    A_t=sqrt(abar_prev)*beta_t/(1-abar_t). For posterior component J,
    innovation = A_t*(E[x0|xt,J]-E[x0|xt])
                 + sqrt(beta_tilde I+A_t^2 Cov[x0|xt,J])*normal.
    All posterior quantities use the fixed Train reference/current label mask.
    Uniforms and normals are supplied so chunking does not change RNG streams.
    """
    if not isinstance(reference, GaussianMixtureReference) or not isinstance(t,int) or isinstance(t,bool) or not 1<=t<=64 or schedule.steps!=64:
        raise ValueError('Registered reference and original64-step integer time required')
    if noisy.ndim!=3 or noisy.shape[-1]!=3 or observed.shape!=noisy.shape[:2] or normal.shape!=noisy.shape or component_uniform.shape!=observed.shape:
        raise ValueError('Exact padded point/mask/innovation shape required')
    if not torch.equal(reference.bar,schedule.bar.to(reference.bar.dtype)) or noisy.dtype!=reference.centers.dtype or noisy.device!=reference.centers.device:
        raise ValueError('Frozen reference/schedule dtype/device equality required')
    if bool(((observed<0)|(observed>reference.classes)).any()) or not bool(torch.isfinite(noisy).all() and torch.isfinite(normal).all() and torch.isfinite(component_uniform).all()) or bool(((component_uniform<0)|(component_uniform>=1)).any()):
        raise ValueError('Finite points/noise, registered labels, uniform in[0,1) required')
    bar=reference.bar[t];previous=schedule.bar[t-1].to(noisy.dtype);beta=schedule.beta[t].to(noisy.dtype);a=bar.sqrt();b=1-bar;A=previous.sqrt()*beta/b;small=beta*(1-previous)/b
    eig=reference.eigenvalues;U=reference.eigenvectors;denominator=bar*eig+b;gain=a*eig/denominator;cov0=eig*b/denominator
    flat=noisy.reshape(-1,3);labels=observed.reshape(-1);eps=normal.reshape(-1,3);uniform=component_uniform.reshape(-1);result=[]
    for start in range(0,len(flat),reference.query_chunk):
        end=start+reference.query_chunk;x=flat[start:end];label=labels[start:end];delta=x[:,None]-a*reference.centers[None];projected=torch.einsum('qmd,mde->qme',delta,U)
        logw=reference.log_component_prior[None]-.5*((projected.square()/denominator[None]).sum(-1)+denominator.log().sum(-1)[None])
        allowed=(label[:,None]==reference.classes)|(label[:,None]==reference.component_class[None]);weights=torch.where(allowed,logw,-torch.inf).softmax(-1)
        m0=reference.centers[None]+torch.einsum('qme,mde->qmd',gain[None]*projected,U);mean0=(weights[:,:,None]*m0).sum(1)
        cdf=weights.cumsum(-1);cdf=cdf/cdf[:,-1:];chosen=(cdf<=uniform[start:end,None]).sum(-1);row=torch.arange(len(x),device=x.device);vectors=U[chosen]
        # Symmetric square root makes the Gaussian draw independent of eigenvector signs.
        projected_noise=torch.einsum('qd,qde->qe',eps[start:end],vectors);standard=(small+A*A*cov0[chosen]).sqrt();gaussian=torch.einsum('qe,qde->qd',projected_noise*standard,vectors)
        result.append(A*(m0[row,chosen]-mean0)+gaussian)
    return torch.cat(result).reshape_as(noisy)


@torch.no_grad()
def sample_reference_innovation(model,schedule,condition,count,generator,component_generator):
    """Same full denoiser/class chain and primary RNG calls; replace centered innovation only."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Only frozen eval models may enter innovation diagnosis')
    device=condition.device;xyz=torch.randn((1,count,3),device=device,generator=generator);labels=torch.full((1,count),model.classes,device=device,dtype=torch.long);valid=torch.ones_like(labels,dtype=torch.bool)
    for t in range(schedule.steps,0,-1):
        ts=torch.tensor([t],device=device);velocity,logits=model(xyz,labels,valid,condition,ts);noise=torch.randn(xyz.shape,device=device,generator=generator)
        uniform=torch.rand(labels.shape,device=device,dtype=xyz.dtype,generator=component_generator)
        innovation=reference_innovation(model.reference,xyz,labels,t,schedule,noise,uniform)
        # This call is exactly the existing full neural mean, including finalt1.
        xyz=schedule.posterior(xyz,velocity,t,torch.zeros_like(noise))+innovation
        labels=absorbing_reverse(labels,logits.softmax(-1),t,model.classes,generator)
    return xyz[0].cpu(),labels[0].cpu()
