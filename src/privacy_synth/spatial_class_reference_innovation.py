"""Class-tilted proxy kernels with identical conditional marginals, different dependence.

This borrows reference within-class distributions, not an exact neural density.
Global centering is essential; class-specific centering changes the coordinate law.
"""
import ast,copy
import torch
from privacy_synth import spatial_set_diffusion as core
from privacy_synth.spatial_mixture_reference import GaussianMixtureReference
from privacy_synth.spatial_reference_innovation_device import reference_schedule_view


@torch.no_grad()
def class_tilted_innovation(reference,noisy,observed,probabilities,proposal,t,schedule,normal,component_uniform,*,coupled):
    if not isinstance(coupled,bool) or not isinstance(reference,GaussianMixtureReference) or not isinstance(t,int) or not 1<=t<=64 or schedule.steps!=64:
        raise ValueError('Explicit coupling/reference/original64 time required')
    if noisy.ndim!=3 or noisy.shape[-1]!=3 or observed.shape!=noisy.shape[:2] or normal.shape!=noisy.shape or proposal.shape!=observed.shape or component_uniform.shape!=observed.shape or probabilities.shape!=(*observed.shape,reference.classes):
        raise ValueError('Exact point/class/innovation shapes required')
    if noisy.dtype!=reference.centers.dtype or noisy.device!=reference.centers.device or not torch.equal(reference.bar,schedule.bar.to(reference.bar.dtype)):
        raise ValueError('Frozen reference proposal view/dtype/device required')
    if not bool(torch.isfinite(noisy).all() and torch.isfinite(normal).all() and torch.isfinite(probabilities).all() and torch.isfinite(component_uniform).all()) or bool((probabilities<0).any()) or bool(((component_uniform<0)|(component_uniform>=1)).any()) or bool(((observed<0)|(observed>reference.classes)).any()) or bool(((proposal<0)|(proposal>=reference.classes)).any()):
        raise ValueError('Finite registered inputs/uniforms/probabilities required')
    total=probabilities.sum(-1,keepdim=True);epsilon=torch.finfo(probabilities.dtype).eps
    if bool((total<=0).any()) or not torch.allclose(total,torch.ones_like(total),rtol=8*epsilon,atol=8*epsilon):raise ValueError('Normalized class probabilities required/no fallback')
    pi=probabilities/total;known=observed<reference.classes;index=torch.where(known,observed,torch.zeros_like(observed));hard=torch.nn.functional.one_hot(index,reference.classes).to(pi.dtype);pi=torch.where(known[:,:,None],hard,pi)
    selected_class=torch.where(known,observed,proposal)
    if coupled and bool((pi.gather(-1,selected_class[:,:,None])<=0).any()):raise ValueError('A zero-probability class cannot be proposed')
    bar=reference.bar[t];a=bar.sqrt();b=1-bar;previous=schedule.bar[t-1].to(noisy.dtype);beta=schedule.beta[t].to(noisy.dtype);A=previous.sqrt()*beta/(1-schedule.bar[t].to(noisy.dtype));small=beta*(1-previous)/(1-schedule.bar[t].to(noisy.dtype))
    eig=reference.eigenvalues;U=reference.eigenvectors;den=bar*eig+b;gain=a*eig/den;cov0=eig*b/den;flat=noisy.reshape(-1,3);classpi=pi.reshape(-1,reference.classes);target=selected_class.reshape(-1);eps=normal.reshape(-1,3);u=component_uniform.reshape(-1);result=[]
    for start in range(0,len(flat),reference.query_chunk):
        end=start+reference.query_chunk;x=flat[start:end];delta=x[:,None]-a*reference.centers[None];projected=torch.einsum('qmd,mde->qme',delta,U);logw=reference.log_component_prior[None]-.5*((projected.square()/den[None]).sum(-1)+den.log().sum(-1)[None]);classes=reference.component_class[None].expand(len(x),-1)
        maximum=torch.full((len(x),reference.classes),-torch.inf,device=x.device,dtype=x.dtype);maximum.scatter_reduce_(1,classes,logw,reduce='amax',include_self=True);unnormalized=(logw-maximum.gather(1,classes)).exp();mass=torch.zeros_like(maximum);mass.scatter_add_(1,classes,unnormalized);within=unnormalized/mass.gather(1,classes)
        tilted=within*classpi[start:end].gather(1,classes);tilted=tilted/tilted.sum(-1,keepdim=True);m0=reference.centers[None]+torch.einsum('qme,mde->qmd',gain[None]*projected,U);global_mean=(tilted[:,:,None]*m0).sum(1)
        weights=torch.where(classes==target[start:end,None],within,0.) if coupled else tilted;weights=weights/weights.sum(-1,keepdim=True);cdf=weights.cumsum(-1);cdf=cdf/cdf[:,-1:];chosen=(cdf<=u[start:end,None]).sum(-1);row=torch.arange(len(x),device=x.device);vectors=U[chosen];projected_noise=torch.einsum('qd,qde->qe',eps[start:end],vectors);gaussian=torch.einsum('qe,qde->qd',projected_noise*(small+A*A*cov0[chosen]).sqrt(),vectors)
        result.append(A*(m0[row,chosen]-global_mean)+gaussian)
    return torch.cat(result).reshape_as(noisy)


# Preserve the original two primary RNG draws and original absorbing transition;
# expose the already-drawn proposal so the coordinate kernel can share its class.
fn=copy.deepcopy(next(n for n in ast.parse(__import__('inspect').getsource(core)).body if isinstance(n,ast.FunctionDef) and n.name=='absorbing_reverse'));baseline=copy.deepcopy(fn);fn.name='category_transition_with_proposal';old_return=copy.deepcopy(fn.body[-1].value);fn.body[-1].value=ast.Tuple(elts=[old_return,ast.Name(id='proposed',ctx=ast.Load())],ctx=ast.Load());inverse=copy.deepcopy(fn);inverse.name='absorbing_reverse';inverse.body[-1].value=copy.deepcopy(inverse.body[-1].value.elts[0])
if ast.dump(inverse)!=ast.dump(baseline):raise PermissionError('Only original category proposal exposure may differ')
namespace=dict(core.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),__file__,'exec'),namespace);category_transition_with_proposal=namespace['category_transition_with_proposal']


@torch.no_grad()
def sample_class_reference_innovation(model,schedule,condition,count,generator,component_generator,*,coupled):
    if not isinstance(coupled,bool) or model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Explicit coupling and frozen eval model required')
    device=condition.device;xyz=torch.randn((1,count,3),device=device,generator=generator);labels=torch.full((1,count),model.classes,device=device,dtype=torch.long);valid=torch.ones_like(labels,dtype=torch.bool);view=reference_schedule_view(model.reference,schedule)
    for t in range(schedule.steps,0,-1):
        velocity,logits=model(xyz,labels,valid,condition,torch.tensor([t],device=device));normal=torch.randn(xyz.shape,device=device,generator=generator);probabilities=logits.softmax(-1);next_labels,proposal=category_transition_with_proposal(labels,probabilities,t,model.classes,generator);uniform=torch.rand(labels.shape,device=device,dtype=xyz.dtype,generator=component_generator)
        innovation=class_tilted_innovation(model.reference,xyz,labels,probabilities,proposal,t,view,normal,uniform,coupled=coupled);xyz=schedule.posterior(xyz,velocity,t,torch.zeros_like(normal))+innovation;labels=next_labels
    return xyz[0].cpu(),labels[0].cpu()
