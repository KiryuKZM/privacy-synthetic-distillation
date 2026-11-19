"""Conditional room-composition reference; no spatial fitting, files, or neural model."""
import math
import time


def features(condition, normalization):
    import numpy as np
    n = condition['count']
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('Positive integer count required')
    size = condition['room_size_native']
    if len(size) != 3 or not all(math.isfinite(v) and v > 0 for v in size):
        raise ValueError('Finite positive room dimensions required')
    raw = np.array([*map(math.log, size), math.log1p(n)])
    mean, std = np.array(normalization['mean']), np.array(normalization['std'])
    if len(mean) != 4 or len(std) != 4 or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Frozen finite four-feature training normalization required')
    return np.r_[1., (raw-mean)/std]


def softmax(logits):
    import numpy as np
    logits = np.asarray(logits)
    x = np.exp(logits-logits.max(axis=-1, keepdims=True))
    return x/x.sum(axis=-1, keepdims=True)


def fit_room_composition(rows, normalization, classes, *, source_role, seconds_max=60):
    import numpy as np
    if source_role != 'inner_train':
        raise ValueError('Composition statistics require inner_train')
    begin = time.monotonic()
    counts = []
    for r in rows:
        if r['count'] != len(r['objects']):
            raise ValueError('All original objects required')
        labels = [o['category_id'] for o in r['objects']]
        if any(isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < classes for k in labels):
            raise ValueError('Registered integer category required')
        counts.append(np.bincount(labels, minlength=classes))
    counts = np.array(counts, dtype=float)
    X = np.array([features(r, normalization) for r in rows])
    pi = (counts/counts.sum(1, keepdims=True)).mean(0)
    if not np.isfinite(pi).all() or (pi <= 0).any():
        raise ValueError('All registered categories need training support, no validation fallback')
    # Exactly one total pseudo-object with the training room-balanced prior.
    probability = (counts+pi)/(counts.sum(1, keepdims=True)+1)
    Y = np.log(probability)
    Y -= Y.mean(1, keepdims=True)
    beta, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    if rank != 5 or len(rows) <= rank:
        raise ValueError('Prospective five-feature regression needs full rank')
    residual = Y-X@beta
    cov = residual.T@residual/(len(rows)-rank)+1e-6*np.eye(classes)
    L = np.linalg.cholesky(cov)
    rng = np.random.default_rng(2026104400)
    noise = rng.normal(size=(len(rows), 2, classes))
    antithetic = np.concatenate([noise, -noise], axis=1).reshape(-1, classes)
    calibration_logits = np.repeat(X@beta, 4, axis=0)+antithetic@L.T
    delta = np.zeros(classes)

    def objective(d):
        z = calibration_logits+d
        top = z.max(1)
        return float((top+np.log(np.exp(z-top[:, None]).sum(1))).mean()-pi@d)

    iterations = 0
    for iteration in range(80):
        if time.monotonic()-begin > seconds_max:
            raise TimeoutError('Bounded composition fit')
        P = softmax(calibration_logits+delta)
        gradient = P.mean(0)-pi
        error = float(np.abs(gradient).max())
        iterations = iteration
        if error < 1e-7:
            break
        # Anchor last logit; softmax otherwise has a constant-shift null direction.
        H = (np.diag(P.mean(0))-P.T@P/len(P))[:-1, :-1]
        direction = np.linalg.solve(H+1e-10*np.eye(classes-1), gradient[:-1])
        descent = float(gradient[:-1]@direction)
        if not math.isfinite(descent) or descent <= 0:
            raise ArithmeticError('Invalid convex calibration direction')
        previous = objective(delta)
        step = 1.
        for _ in range(24):
            trial = delta.copy()
            trial[:-1] -= step*direction
            if objective(trial) <= previous-1e-4*step*descent:
                delta = trial
                break
            step *= .5
        else:
            raise ArithmeticError('Calibration line search failed; preserve, no tuning')
    else:
        raise ArithmeticError('Calibration did not converge within80 iterations')
    delta -= delta.mean()
    final_error = float(np.abs(softmax(calibration_logits+delta).mean(0)-pi).max())
    if final_error >= 1e-7:
        raise ArithmeticError('Training marginal calibration tolerance')
    return dict(regression=beta.tolist(), residual_covariance=cov.tolist(), residual_cholesky=L.tolist(),
        intercept_adjustment=delta.tolist(), prior=pi.tolist(), classes=classes, normalization=normalization,
        source_role=source_role, training_rooms=len(rows), regression_rank=int(rank),
        one_total_prior_object=True, covariance_ridge=1e-6, calibration_seed=2026104400,
        calibration_vectors=4*len(rows), antithetic_vectors_per_room=4,
        calibration_iterations=iterations, calibration_max_marginal_error=final_error,
        calibration_scope='Fixed numerical integration over TRAIN conditions/no validation tuning; not a raw-scene evaluation',
        fit_seconds=time.monotonic()-begin)


def sample_categories(condition, parameters, seed):
    import numpy as np
    x = features(condition, parameters['normalization'])
    rng = np.random.default_rng(seed)
    logits = x@np.array(parameters['regression'])+np.array(parameters['residual_cholesky'])@rng.normal(size=parameters['classes'])+parameters['intercept_adjustment']
    p = softmax(logits)
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError('Invalid composition probability, no fallback')
    return [int(k) for k in rng.choice(parameters['classes'], size=condition['count'], p=p)]
