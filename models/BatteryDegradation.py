import numpy as np
from sklearn.neighbors import NearestNeighbors


def eta(socl, soch, N):
    socl, soch = socl / 100., soch / 100.

    if 0 <= soch <= 1 and 0 <= soch <= 1:
        if soch > socl:
            SR_m = (soch + socl) / 2.
            SR_l = (soch - socl)
            tabla = np.array([[0.5, 1, 1.00000000], [0.625, 0.75, 1.00000266], [0.375, 0.75, 1.00001860],
                              [0.75, 0.5, 0.99999203], [0.5, 0.5, 1.00001521], [0.25, 0.5, 1.00002874],
                              [0.875, 0.25, 1.00002146], [0.625, 0.25, 1.00000881], [0.5, 0.25, 1.00000620],
                              [0.375, 0.25, 1.00003347], [0.125, 0.25, 1.00004184]])

            x = tabla[:, 0:2]
            y = np.array([[SR_m, SR_l]])
            nbrs = NearestNeighbors(n_neighbors=3).fit(x)
            D, IDX = nbrs.kneighbors(y)

            W1 = 1 / (1 + (D[0, 0] / D[0, 1]) + (D[0, 0] / D[0, 2]))
            W2 = 1 / (1 + (D[0, 1] / D[0, 0]) + (D[0, 1] / D[0, 2]))
            W3 = 1 / (1 + (D[0, 2] / D[0, 0]) + (D[0, 2] / D[0, 1]))

            eta_f = W1*(tabla[IDX[0, 0], 2]) + W2*(tabla[IDX[0, 1], 2]) + W3*(tabla[IDX[0, 2], 2])
            eta_bat = (0.8 ** (1 / N)) * eta_f
        else:
            eta_bat = 1
        OUT = eta_bat
    else:
        OUT = 0
    return OUT


if __name__ == '__main__':
    print(eta(40, 80, 1000))
