import numpy as np


def gamma(tSeries, k, nNodes):
    """
    Calculates gamma vector and delta at instant k

    :param tSeries: time series ordered in ascendant fashion according to global counter. Each element of tseries has
                    the form [timeStamp, occurrenceNode, eventType, occurrenceVehicle]
    :param k: instant to calculate gamma
    :param nNodes: amount of nodes in the network, including depot
    :return: tuple in the form (gammaVector, delta)
    """
    g = np.zeros((nNodes, 1))
    occurrenceNode = tSeries[k][1]
    delta = int(tSeries[k][2])
    g[occurrenceNode] = delta
    return g, delta
