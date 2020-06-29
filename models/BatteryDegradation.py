import numpy as np


def eta(socl, soch, N):
    socl, soch = socl/100., soch/100.

    if 0 <= soch <= 1 and 0 <= soch <= 1:
        if soch > socl:
            SR_m = (soch+socl)/2.
            SR_l = (soch-socl)

    return
