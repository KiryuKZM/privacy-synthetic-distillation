"""Frozen Train-only finite room composition, plus the existing neural residual.

The numerical intercept calibration targets training room-balanced class mass.
It does not refit the old regression/covariance or use a validation statistic.
"""
import math,time
import torch
from torch import nn
from privacy_synth.spatial_composition import features,softmax
from privacy_synth.spatial_mixture_reference import GaussianMixtureReference
from privacy_synth.spatial_mixture_residual import MixtureReferenceSetDenoiser
from privacy_synth.spatial_joint_composition_reference import FiniteCompositionReference


def calibrate_finite_composition(rows,parameters,latent_points,*,source_role,seconds_max=60):
    import numpy as np
    if source_role!='inner_train' or parameters['source_role']!='inner_train' or len(rows)!=parameters['training_rooms']:
        raise ValueError('All original inner_train conditions required')
    begin=time.monotonic();K=parameters['classes'];Z=np.asarray(latent_points,dtype=float)
    if Z.ndim!=2 or Z.shape[1]!=K or len(Z)<1 or not np.isfinite(Z).all():raise ValueError('Explicit finite latent integration points required')
    X=np.array([features(r,parameters['normalization']) for r in rows]);regression=np.array(parameters['regression']);L=np.array(parameters['residual_cholesky']);target=np.array(parameters['prior'])
    if regression.shape!=(5,K) or L.shape!=(K,K) or target.shape!=(K,) or (target<=0).any() or not np.isfinite(target).all() or abs(target.sum()-1)>1e-10:raise ValueError('Frozen supported composition parameters required')
    logits=(X@regression)[:,None,:]+(Z@L.T)[None,:,:];logits=logits.reshape(-1,K);delta=np.array(parameters['intercept_adjustment'],dtype=float)
    def objective(d):
        z=logits+d;top=z.max(1);return float((top+np.log(np.exp(z-top[:,None]).sum(1))).mean()-target@d)
    for iteration in range(80):
        if time.monotonic()-begin>seconds_max:raise TimeoutError('Finite60s Train calibration')
        P=softmax(logits+delta);gradient=P.mean(0)-target
        if float(np.abs(gradient).max())<1e-7:break
        H=(np.diag(P.mean(0))-P.T@P/len(P))[:-1,:-1];direction=np.linalg.solve(H+1e-10*np.eye(K-1),gradient[:-1]);descent=float(gradient[:-1]@direction)
        if not math.isfinite(descent) or descent<=0:raise ArithmeticError('Invalid numerical calibration direction')
        previous=objective(delta);step=1.
        for _ in range(24):
            trial=delta.copy();trial[:-1]-=step*direction
            if objective(trial)<=previous-1e-4*step*descent:delta=trial;break
            step*=.5
        else:raise ArithmeticError('Numerical calibration line search failed/no tuning')
    else:raise ArithmeticError('Numerical calibration80 limit/no tuning')
    delta-=delta.mean();error=float(np.abs(softmax(logits+delta).mean(0)-target).max())
    if error>=1e-7:raise ArithmeticError('Frozen training marginal target not reached')
    result=dict(parameters);result.update(intercept_adjustment=delta.tolist(),legacy_intercept_adjustment=parameters['intercept_adjustment'],finite_latent_points=Z.tolist(),finite_components=len(Z),finite_calibration_vectors=len(logits),finite_calibration_iterations=iteration,finite_calibration_max_marginal_error=error,finite_calibration_seconds=time.monotonic()-begin,scope='Normalized finite conditional composition prior; not exact logistic-normal integration. Only intercept recalibrated on ALL Train conditions; regression/covariance/spatial reference unchanged.')
    return result


class FrozenRoomCompositionReference(nn.Module):
    def __init__(self,point_reference,parameters):
        super().__init__();self.joint=FiniteCompositionReference(point_reference);self.classes=self.joint.classes
        if parameters['source_role']!='inner_train' or parameters['classes']!=self.classes:raise ValueError('Matched frozen training composition required')
        dtype=point_reference.centers.dtype
        self.register_buffer('regression',torch.tensor(parameters['regression'],dtype=dtype))
        self.register_buffer('latent_points',torch.tensor(parameters['finite_latent_points'],dtype=dtype))
        self.register_buffer('cholesky',torch.tensor(parameters['residual_cholesky'],dtype=dtype))
        self.register_buffer('intercept',torch.tensor(parameters['intercept_adjustment'],dtype=dtype))
        if self.regression.shape!=(5,self.classes) or self.cholesky.shape!=(self.classes,self.classes) or self.latent_points.ndim!=2 or self.latent_points.shape[1]!=self.classes or self.intercept.shape!=(self.classes,):raise ValueError('Registered finite composition shapes required')
        if not all(bool(torch.isfinite(v).all()) for v in [self.regression,self.latent_points,self.cholesky,self.intercept]):raise ValueError('Finite frozen composition required')

    @property
    def bar(self):return self.joint.point.bar

    def mixing_logits(self,condition):
        if condition.shape!=(len(condition),4) or condition.dtype!=self.regression.dtype or condition.device!=self.regression.device or not bool(torch.isfinite(condition).all()):raise ValueError('Original standardized four conditions required')
        X=torch.cat([torch.ones_like(condition[:,:1]),condition],-1)
        return (X@self.regression)[:,None,:]+(self.latent_points@self.cholesky.T)[None,:,:]+self.intercept

    def forward(self,noisy,observed,valid,condition,t):
        return self.joint(noisy,observed,t,valid,self.mixing_logits(condition))


class RoomCompositionSetDenoiser(MixtureReferenceSetDenoiser):
    """Only the analytic reference changes; paired network init/RNG are preserved."""
    def __init__(self,gaussian_parameters,mixture_parameters,prior,composition_parameters):
        super().__init__(gaussian_parameters,mixture_parameters,prior)
        self.reference=FrozenRoomCompositionReference(self.reference,composition_parameters)

    def forward(self,noisy,observed,valid,condition,t):
        velocity,logits=self.reference(noisy,observed,valid,condition,t)
        residual_v,residual_logits=self.network(noisy,observed,valid,condition,t)
        return velocity+residual_v,logits+residual_logits


class RoomCompositionPriorOnly(nn.Module):
    def __init__(self,mixture_parameters,prior,composition_parameters):
        super().__init__();point=GaussianMixtureReference(mixture_parameters,prior,query_chunk=64)
        self.reference=FrozenRoomCompositionReference(point,composition_parameters);self.classes=self.reference.classes

    def forward(self,noisy,observed,valid,condition,t):
        return self.reference(noisy,observed,valid,condition,t)
