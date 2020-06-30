import numpy as np


def eta(socl, soch, N):
    socl, soch = socl/100., soch/100.

    if 0 <= soch <= 1 and 0 <= soch <= 1:
        if soch > socl:
            SR_m = (soch+socl)/2.
            SR_l = (soch-socl)
            tabla = [[0.5, 1, 1.00000000], [0.625, 0.75, 1.00000266], [0.375, 0.75, 1.00001860],
                     [0.75, 0.5, 0.99999203], [0.5, 0.5, 1.00001521], [0.25, 0.5, 1.00002874],
                     [0.875, 0.25, 1.00002146], [0.625, 0.25, 1.00000881], [0.5, 0.25, 1.00000620],
                     [0.375, 0.25, 1.00003347], [0.125, 0.25, 1.00004184]]
    return
