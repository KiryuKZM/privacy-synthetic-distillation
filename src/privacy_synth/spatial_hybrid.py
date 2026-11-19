"""Compose a category-set generator with the unchanged Fit Gaussian coordinate component."""
import math

def draw_coordinates(condition, parameters, normalization, seed, categories=None):
    import numpy as np
    count=condition['count'];K=len(parameters)
    probability=np.array([p['n'] for p in parameters],float);probability/=probability.sum()
    rng=np.random.default_rng(seed)
    # Consume the original baseline's categorical draw also for hybrids. This
    # preserves common coordinate random numbers; no selected coordinate rerolls.
    original=rng.choice(K,size=count,p=probability)
    labels=original if categories is None else categories
    if len(labels)!=count or any(isinstance(x,bool) or not isinstance(x,(int,np.integer)) or not 0<=int(x)<K for x in labels):
        raise ValueError('Complete registered category set required; no refill')
    objects=[]
    for label in labels:
        label=int(label);p=parameters[label];z=rng.multivariate_normal(p['mean'],p['covariance'])
        xyz=[(math.sinh(v)*s+c)*room for v,s,c,room in zip(z,normalization['IQR'],normalization['center'],condition['room_size_native'])]
        if not all(math.isfinite(v) for v in z.tolist()+xyz):raise ValueError('Nonfinite sample, no clipping/replacement')
        objects.append(dict(category_id=label,position_native=xyz,position_latent=z.tolist()))
    return objects
