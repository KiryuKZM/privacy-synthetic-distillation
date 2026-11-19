"""Fit-only class mixtures preserving the Gaussian reference's first two moments."""
import math

def fit_class(points, prior):
    import numpy as np
    a=np.array(points,dtype=float);mean=np.array(prior['mean']);target=np.array(prior['covariance'])
    if len(a)!=prior['n'] or len(a)<2 or not np.isfinite(a).all():raise ValueError('Full finite class pool required')
    if not np.allclose(a.mean(0),mean,rtol=0,atol=1e-12):raise ValueError('Original reference mean identity')
    population=(a-mean).T@(a-mean)/len(a);h=len(a)**(-1/7);beta=1/math.sqrt(1+h*h)
    noise=target-beta*beta*population
    if np.linalg.eigvalsh(noise).min()<=0:raise ValueError('Invalid noise covariance; no fallback or clipping')
    reconstructed=beta*beta*population+noise
    if not np.allclose(reconstructed,target,rtol=0,atol=1e-12):raise AssertionError('Mixture covariance identity')
    return dict(n=len(a),mean=mean.tolist(),target_covariance=target.tolist(),centers=a.tolist(),beta=beta,noise_covariance=noise.tolist(),scott_factor=h)

def draw_coordinates(condition, parameters, normalization, seed, categories=None, gaussian_limit=False):
    import numpy as np
    if gaussian_limit:
        from privacy_synth.spatial_hybrid import draw_coordinates as original
        prior=[dict(n=p['n'],mean=p['mean'],covariance=p['target_covariance']) for p in parameters]
        return original(condition,prior,normalization,seed,categories)
    count=condition['count'];K=len(parameters);prob=np.array([p['n'] for p in parameters],float);prob/=prob.sum()
    rng=np.random.default_rng(seed);component_rng=np.random.default_rng(seed+901000000);base=rng.choice(K,size=count,p=prob);labels=base if categories is None else categories
    if len(labels)!=count or any(isinstance(x,bool) or not isinstance(x,(int,np.integer)) or not 0<=int(x)<K for x in labels):raise ValueError('Complete registered integer category set required')
    out=[]
    for label in labels:
        label=int(label);p=parameters[label];j=int(component_rng.integers(p['n']));mu=np.array(p['mean']);center=mu+p['beta']*(np.array(p['centers'][j])-mu)
        z=rng.multivariate_normal(center,p['noise_covariance']);xyz=[(math.sinh(v)*s+c)*room for v,s,c,room in zip(z,normalization['IQR'],normalization['center'],condition['room_size_native'])]
        if not all(math.isfinite(v) for v in z.tolist()+xyz):raise ValueError('Nonfinite mixture output; no clipping/refill')
        out.append(dict(category_id=label,position_native=xyz,position_latent=z.tolist()))
    return out
