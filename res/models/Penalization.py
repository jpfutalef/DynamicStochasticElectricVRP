import math


def normal_cdf(x, mu=0., sigma=1.0):
    if sigma:
        q = math.erf((x - mu) / (math.sqrt(2.0) * sigma))
        return (1.0 + q) / 2.0
    elif x - mu:
        return 0.
    return 1.


def penalization_deterministic(x: float, y: float, **kwargs):
    return quadratic(x, y, **kwargs)


def penalization_stochastic(cdf: float, const_prob: float, **kwargs):
    if cdf > const_prob:
        return 0.  # Constraint is satisfied
    return linear(const_prob, cdf, **kwargs)


def quadratic(x: float, y: float, w=1.0, c=0):
    return w * (x - y) ** 2 + c


def magnitude(x: float, y: float, w=1.0, c=0):
    return w * abs(x - y) + c


def linear(x: float, y: float, w=1.0, c=0):
    return w * (x - y) + c
