"""CPU-only frozen-score covariance diagnostics; never constructs a sampling kernel."""
import torch


@torch.no_grad()
def reference_reverse_covariance(reference,noisy,observed,t,schedule):
    """Full small-room block covariance of the independent analytic reference.

    Includes both within-component and between-component clean uncertainty.
    Float64 arithmetic uses the stored reference/schedule values without repair.
    """
    if noisy.device.type!='cpu' or noisy.ndim!=3 or noisy.shape[0]!=1 or noisy.shape[-1]!=3 or observed.shape!=noisy.shape[:2]:
        raise ValueError('Single CPU room with unchanged point/mask shape required')
    if not isinstance(t,int) or isinstance(t,bool) or not 1<=t<=64 or schedule.steps!=64:
        raise ValueError('Original64 integer time required')
    if bool(((observed<0)|(observed>reference.classes)).any()) or not bool(torch.isfinite(noisy).all()):
        raise ValueError('Finite registered known/MASK inputs required')
    x=noisy[0].double();label=observed[0];bar=reference.bar[t].double();a=bar.sqrt();b=1-bar;centers=reference.centers.double();eig=reference.eigenvalues.double();U=reference.eigenvectors.double();den=bar*eig+b
    delta=x[:,None]-a*centers[None];projected=torch.einsum('nmd,mde->nme',delta,U);logw=reference.log_component_prior.double()[None]-.5*((projected.square()/den[None]).sum(-1)+den.log().sum(-1)[None]);allowed=(label[:,None]==reference.classes)|(label[:,None]==reference.component_class[None]);weights=torch.where(allowed,logw,-torch.inf).softmax(-1)
    mean0=centers[None]+torch.einsum('nme,mde->nmd',(a*eig/den)[None]*projected,U);mean=(weights[:,:,None]*mean0).sum(1);dev=mean0-mean[:,None];component_cov=torch.einsum('mde,me,mfe->mdf',U,eig*b/den,U)
    cov0=torch.einsum('nm,mdf->ndf',weights,component_cov)+torch.einsum('nm,nmd,nmf->ndf',weights,dev,dev)
    beta=schedule.beta[t].double();previous=schedule.bar[t-1].double();A=previous.sqrt()*beta/(1-schedule.bar[t].double());small=beta*(1-previous)/(1-schedule.bar[t].double());blocks=small*torch.eye(3,dtype=torch.float64)[None]+A*A*cov0
    return torch.block_diag(*blocks.unbind(0))


def residual_input_jacobian(model,noisy,observed,condition,t):
    """Input derivatives only; frozen parameter grads/state remain untouched."""
    if noisy.device.type!='cpu' or noisy.shape[0]!=1 or noisy.ndim!=3 or noisy.shape[-1]!=3 or not 1<=noisy.shape[1]<=8:
        raise ValueError('Prospective CPU diagnostic is bounded to complete rooms of at most8 objects')
    if model.training or any(p.requires_grad for p in model.parameters()) or not 1<=t<=64:
        raise ValueError('Frozen eval model/original trained time required')
    if t==64 and bool((observed!=model.classes).any()):
        raise ValueError('Neural t64 input must be all MASK, as in training')
    valid=torch.ones_like(observed,dtype=torch.bool);time=torch.tensor([t]);x=noisy.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        def velocity(value):return model.network(value,observed,valid,condition,time)[0]
        jac=torch.autograd.functional.jacobian(velocity,x,create_graph=False,strict=False,vectorize=False)
    result=jac.reshape(noisy.numel(),noisy.numel()).detach().double()
    if not bool(torch.isfinite(result).all()) or any(p.grad is not None for p in model.parameters()):
        raise AssertionError('Finite input-only Jacobian and no parameter gradients required')
    return result


def score_implied_covariance(reference_covariance,residual_jacobian,t,schedule):
    """Tweedie consistency formula under an exact joint-score assumption only.

    Raw matrix is returned; no symmetrization, eigenvalue clipping or PSD repair.
    """
    beta=schedule.beta[t].double();alpha=schedule.alpha[t].double();bar=schedule.bar[t].double();factor=(beta.square()/alpha)*(bar/(1-bar)).sqrt();correction=-factor*residual_jacobian.double()
    return reference_covariance.double()+correction,correction


@torch.no_grad()
def covariance_diagnostics(reference_covariance,implied,correction):
    """Symmetric-part eigenvalues are diagnostics, never a modified proposal."""
    if implied.shape[0]!=implied.shape[1] or implied.shape[0]%3:raise ValueError('Full xyz-by-object square matrix required')
    scale=float(torch.linalg.matrix_norm(reference_covariance));delta_scale=float(torch.linalg.matrix_norm(correction));implied_scale=float(torch.linalg.matrix_norm(implied));cross=correction.clone()
    for i in range(0,len(cross),3):cross[i:i+3,i:i+3]=0
    symmetric=(implied+implied.T)/2;skew=(implied-implied.T)/2;eigenvalues=torch.linalg.eigvalsh(symmetric);reference_eigenvalues=torch.linalg.eigvalsh(reference_covariance)
    return dict(dimension=len(implied),correction_Frobenius_over_reference=float(torch.linalg.matrix_norm(correction))/scale,cross_object_correction_Frobenius_over_reference=float(torch.linalg.matrix_norm(cross))/scale,cross_object_fraction_of_correction=None if delta_scale==0 else float(torch.linalg.matrix_norm(cross))/delta_scale,
        skew_Frobenius_over_reference=float(torch.linalg.matrix_norm(skew))/scale,relative_symmetry_defect=None if implied_scale==0 else float(torch.linalg.matrix_norm(implied-implied.T))/implied_scale,reference_min_eigenvalue=float(reference_eigenvalues.min()),symmetric_part_min_eigenvalue=float(eigenvalues.min()),symmetric_part_negative_eigenvalues=int((eigenvalues<0).sum()),matrix_used_for_sampling=False)
