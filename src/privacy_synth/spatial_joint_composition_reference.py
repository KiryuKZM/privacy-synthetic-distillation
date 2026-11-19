"""Finite shared room-composition posterior over frozen class spatial mixtures.

This reference captures a mixture-of-iid subset of room dependence. It neither
models arbitrary exclusion constraints nor makes a factorized reverse sampler
exact. No fitting, randomness, files, new neural parameters, or probability floor.
The returned log_evidence omits common Gaussian and MASK-pattern constants; it
is for posterior normalization/score, never a complete normalized NLL. Only
valid positions of log_class represent posterior probabilities after exp.
Composition/latent logits must be fixed given external conditions, not estimated
again from the same noisy query while claiming this score formula.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from privacy_synth.spatial_mixture_reference import GaussianMixtureReference


def composition_posterior(log_likelihood, class_mean0, composition_logits,
                          observed, valid, latent_logits=None):
    """Return one coherent joint posterior; MASK is the final class index."""
    if log_likelihood.ndim != 3:
        raise ValueError('B,N,K class likelihood required')
    B,N,K=log_likelihood.shape
    if class_mean0.shape!=(B,N,K,3) or composition_logits.ndim!=3 or composition_logits.shape[0]!=B or composition_logits.shape[2]!=K or observed.shape!=(B,N) or valid.shape!=(B,N):
        raise ValueError('Matched explicit batch/object/class/latent shapes required')
    H=composition_logits.shape[1]
    if min(B,N,K,H)<1 or valid.dtype!=torch.bool or observed.dtype!=torch.long or not bool(valid.any(-1).all()):
        raise ValueError('Nonempty rooms and registered label/valid types required')
    if bool(((observed<0)|(observed>K)).any()):
        raise ValueError('Only known classes or explicit MASK')
    tensors=[class_mean0,composition_logits]
    if any(v.dtype!=log_likelihood.dtype or v.device!=log_likelihood.device for v in tensors) or observed.device!=log_likelihood.device or valid.device!=log_likelihood.device:
        raise ValueError('Same dtype/device required')
    if not bool(torch.isfinite(log_likelihood).all()) or not all(bool(torch.isfinite(v).all()) for v in tensors):
        raise ValueError('Finite supported log densities/parameters required')
    if latent_logits is None:
        log_h=torch.full((B,H),-math.log(H),device=log_likelihood.device,dtype=log_likelihood.dtype)
    else:
        if latent_logits.shape!=(B,H) or latent_logits.dtype!=log_likelihood.dtype or latent_logits.device!=log_likelihood.device or not bool(torch.isfinite(latent_logits).all()):
            raise ValueError('Finite B,H latent logits required')
        log_h=F.log_softmax(latent_logits,-1)
    log_pi=F.log_softmax(composition_logits,-1)
    log_joint=log_pi[:,:,None,:]+log_likelihood[:,None,:,:]
    log_mask_evidence=torch.logsumexp(log_joint,-1)
    known=observed<K;index=observed.clamp_max(K-1)
    log_known_evidence=log_joint.gather(-1,index[:,None,:,None].expand(B,H,N,1)).squeeze(-1)
    log_object_evidence=torch.where(known[:,None,:],log_known_evidence,log_mask_evidence)
    log_room=log_h+torch.where(valid[:,None,:],log_object_evidence,0.).sum(-1)
    log_evidence=torch.logsumexp(log_room,-1)
    log_posterior_h=log_room-log_evidence[:,None]
    log_class_if_h=F.log_softmax(log_joint,-1)
    log_class=torch.logsumexp(log_posterior_h[:,:,None,None]+log_class_if_h,1)
    hard=F.one_hot(index,K).to(log_likelihood.dtype)
    log_hard=torch.where(hard.bool(),0.,-torch.inf)
    log_class=torch.where(known[:,:,None],log_hard,log_class)
    mean0=(log_class.exp()[:,:,:,None]*class_mean0).sum(2)
    mean0=torch.where(valid[:,:,None],mean0,0.)
    log_class=torch.where(valid[:,:,None],log_class,0.)
    return dict(mean0=mean0,log_class=log_class,log_posterior_h=log_posterior_h,
                log_evidence=log_evidence)


class FiniteCompositionReference(nn.Module):
    """Whole-room posterior, reusing the immutable point-reference buffers."""
    def __init__(self, point_reference):
        super().__init__()
        if not isinstance(point_reference,GaussianMixtureReference):
            raise TypeError('Frozen registered GaussianMixtureReference required')
        self.point=point_reference
        self.classes=point_reference.classes
        class_log_prior=torch.stack([torch.logsumexp(point_reference.log_component_prior[point_reference.component_class==k],0) for k in range(self.classes)])
        self.register_buffer('class_log_prior',class_log_prior)

    @torch.no_grad()
    def class_channels(self,noisy,t):
        p=self.point
        if noisy.ndim!=3 or noisy.shape[-1]!=3 or t.shape!=(len(noisy),) or noisy.device!=p.centers.device or noisy.dtype!=p.centers.dtype or t.device!=noisy.device or t.dtype!=torch.long or bool(((t<1)|(t>64)).any()) or not bool(torch.isfinite(noisy).all()):
            raise ValueError('Finite B,N,3 and original64 supported time/device required')
        B,N,_=noisy.shape;flat=noisy.reshape(-1,3);bars=p.bar[t][:,None].expand(B,N).reshape(-1);likelihood=[];means=[]
        for start in range(0,len(flat),p.query_chunk):
            x=flat[start:start+p.query_chunk];bar=bars[start:start+p.query_chunk];a=bar.sqrt();den=bar[:,None,None]*p.eigenvalues[None]+(1-bar)[:,None,None]
            delta=x[:,None]-a[:,None,None]*p.centers[None];projected=torch.einsum('qmd,mde->qme',delta,p.eigenvectors)
            logw=p.log_component_prior[None]-.5*((projected.square()/den).sum(-1)+den.log().sum(-1));index=p.component_class[None].expand(len(x),-1)
            maximum=torch.full((len(x),self.classes),-torch.inf,device=x.device,dtype=x.dtype);maximum.scatter_reduce_(1,index,logw,reduce='amax',include_self=True)
            weight=(logw-maximum.gather(1,index)).exp();mass=torch.zeros_like(maximum);mass.scatter_add_(1,index,weight)
            gain=a[:,None,None]*p.eigenvalues[None]/den;mean0=p.centers[None]+torch.einsum('qme,mde->qmd',gain*projected,p.eigenvectors)
            total=torch.zeros(len(x),self.classes,3,device=x.device,dtype=x.dtype);total.scatter_add_(1,index[:,:,None].expand(-1,-1,3),weight[:,:,None]*mean0)
            likelihood.append(maximum+mass.log()-self.class_log_prior[None]);means.append(total/mass[:,:,None])
        return torch.cat(likelihood).reshape(B,N,self.classes),torch.cat(means).reshape(B,N,self.classes,3)

    @torch.no_grad()
    def forward(self,noisy,observed,t,valid,composition_logits,latent_logits=None,*,return_details=False):
        if observed.shape!=noisy.shape[:2] or valid.shape!=observed.shape:
            raise ValueError('Matched valid/known-MASK objects required')
        if bool(((t==64)[:,None] & valid & (observed<self.classes)).any()):
            raise ValueError('Known class at fully MASK terminal has zero forward probability')
        log_likelihood,mean_by_class=self.class_channels(noisy,t)
        post=composition_posterior(log_likelihood,mean_by_class,composition_logits,observed,valid,latent_logits)
        bar=self.point.bar[t][:,None,None]
        velocity=(bar.sqrt()*noisy-post['mean0'])/(1-bar).sqrt()
        velocity=torch.where(valid[:,:,None],velocity,0.)
        return (velocity,post['log_class'],post) if return_details else (velocity,post['log_class'])
