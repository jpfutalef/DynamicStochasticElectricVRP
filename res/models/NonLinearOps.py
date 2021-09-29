import numpy as np


def saturate(val, min_val=-np.infty, max_val=np.infty):
    if val > max_val:
        return max_val
    elif val < min_val:
        return min_val
    return val


def fix_tod(tod: float):
    while tod > 86400:
        tod -= 86400
    return tod
