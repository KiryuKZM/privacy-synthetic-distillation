"""Invertible class-coordinate preprocessing; no model, file I/O, or sampling."""
import copy
import math


def fit_normalization(rows, category_vocab, *, source_role):
    import numpy as np
    if source_role != 'inner_train':
        raise ValueError('Normalization statistics require inner_train')
    q = np.array([np.array(o['position_native']) / r['room_size_native'] for r in rows for o in r['objects']])
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
        raise ValueError('Full finite native coordinates required')
    center = np.median(q, axis=0)
    scale = np.maximum(np.quantile(q, .75, axis=0)-np.quantile(q, .25, axis=0), 1e-6)
    return dict(category_vocab=list(category_vocab), center=center.tolist(), IQR=scale.tolist(),
                fit_only=True, source_role=source_role, objects=len(q), transform='asinh((q-median)/IQR), no clipping')


def global_rows(rows, normalization):
    """Derive new coordinates from native values, ignoring any prior latent data."""
    import numpy as np
    out = copy.deepcopy(rows)
    center, scale = np.array(normalization['center']), np.array(normalization['IQR'])
    for r in out:
        if r['count'] != len(r['objects']) or r['count'] < 1 or min(r['room_size_native']) <= 0:
            raise ValueError('Original nonempty complete room required')
        for o in r['objects']:
            q = np.array(o['position_native']) / r['room_size_native']
            z = np.arcsinh((q-center)/scale)
            if not np.isfinite(z).all():
                raise ValueError('Nonfinite global transform')
            o['position_latent'] = z.tolist()
    return out


def fit_whitening(rows, classes, *, source_role):
    import numpy as np
    if source_role != 'inner_train':
        raise ValueError('Whitening statistics require inner_train')
    pools = [[] for _ in range(classes)]
    for r in rows:
        for o in r['objects']:
            pools[o['category_id']].append(o['position_latent'])
    parameters = []
    for pool in pools:
        if len(pool) < 2:
            raise ValueError('Class needs >=2 training objects; no silent fallback')
        a = np.array(pool)
        cov = np.cov(a, rowvar=False) + 1e-6*np.eye(3)
        if not np.isfinite(a).all() or np.linalg.eigvalsh(cov).min() <= 0:
            raise ValueError('Finite SPD covariance required')
        parameters.append(dict(n=len(a), mean=a.mean(0).tolist(), covariance=cov.tolist(),
                               cholesky=np.linalg.cholesky(cov).tolist()))
    return dict(parameters=parameters, source_role=source_role, ridge=1e-6,
                transform='white=solve(L,z-mean), inverse=L@white+mean; no clipping')


def encode_rows(rows, whitening):
    import numpy as np
    out = copy.deepcopy(rows)
    for r in out:
        for o in r['objects']:
            p = whitening['parameters'][o['category_id']]
            z = np.linalg.solve(np.array(p['cholesky']), np.array(o['position_latent'])-p['mean'])
            if not np.isfinite(z).all():
                raise ValueError('Nonfinite whitening transform')
            o['position_latent'] = z.tolist()
    return out


def decode_point(white, label, whitening, normalization, room_size):
    import numpy as np
    if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(whitening['parameters']):
        raise ValueError('Registered final class required, no MASK/fallback')
    p = whitening['parameters'][label]
    z = np.array(p['cholesky']) @ np.array(white) + p['mean']
    xyz = [(math.sinh(v)*s+c)*room for v, s, c, room in zip(z, normalization['IQR'], normalization['center'], room_size)]
    if not np.isfinite(z).all() or not all(math.isfinite(v) for v in xyz):
        raise ValueError('Nonfinite inverse, no clipping/refill')
    return dict(category_id=label, position_latent=z.tolist(), position_native=xyz,
                position_whitened=list(white))
