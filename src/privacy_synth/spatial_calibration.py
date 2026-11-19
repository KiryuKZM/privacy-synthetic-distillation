"""Room-balanced positive axis-affine calibration without clipping or rotation."""
import math


def room_moments(sets):
    means, squares = [], []
    for points in sets:
        if not points or any(len(p) != 3 or not all(math.isfinite(x) for x in p) for p in points):
            raise ValueError('Every calibration room must contain finite full 3D points')
        means.append([sum(p[j] for p in points) / len(points) for j in range(3)])
        squares.append([sum(p[j] ** 2 for p in points) / len(points) for j in range(3)])
    if not means:
        raise ValueError('No calibration rooms')
    mean = [sum(v[j] for v in means) / len(means) for j in range(3)]
    variance = [sum(v[j] for v in squares) / len(means) - mean[j] ** 2 for j in range(3)]
    if any(v <= 1e-12 or not math.isfinite(v) for v in variance):
        raise ValueError('Degenerate variance; no invented fallback scale')
    return dict(mean=mean, std=[math.sqrt(v) for v in variance],
                weight='equal rooms; equal objects within room')


def fit_axis_affine(generated, target):
    for moments in (generated, target):
        if len(moments['mean']) != 3 or len(moments['std']) != 3:
            raise ValueError('Exactly three axes required')
        if any(not math.isfinite(x) for x in moments['mean'] + moments['std']):
            raise ValueError('Nonfinite moments')
        if any(s <= 1e-12 for s in moments['std']):
            raise ValueError('Degenerate scale')
    scale = [a / b for a, b in zip(target['std'], generated['std'])]
    shift = [a - s * b for a, b, s in zip(target['mean'], generated['mean'], scale)]
    return dict(scale=scale, shift=shift)


def transform(points, parameters):
    scale, shift = parameters['scale'], parameters['shift']
    if len(scale) != 3 or len(shift) != 3 or any(s <= 0 or not math.isfinite(s) for s in scale):
        raise ValueError('Positive finite 3-axis scales required')
    if not all(math.isfinite(x) for x in shift):
        raise ValueError('Finite shifts required')
    if any(len(p) != 3 or not all(math.isfinite(x) for x in p) for p in points):
        raise ValueError('Finite full 3D points required')
    out = [[s * v + b for s, v, b in zip(scale, p, shift)] for p in points]
    if any(not all(math.isfinite(x) for x in p) for p in out):
        raise ValueError('Affine overflow retained as invalid, no clipping')
    return out
