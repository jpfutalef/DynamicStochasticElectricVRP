import numpy as np


def gamma(tseries, k, nNodes):
    """
    Calculates gamma vector and delta at instant k

    :param tseries: time series ordered in ascendant fashion according to global counter. Each element of tseries has
                    the form [timeStamp, occurrenceNode, eventType, occurrenceVehicle]
    :param k: instant to calculate gamma
    :param nNodes: amount of nodes in the network, including depot
    :return: tuple in the form (gammaVector, delta)
    """
    g = np.zeros((nNodes, 1))
    occurrenceNode = tseries[k][1]
    delta = int(tseries[k][2])
    g[occurrenceNode] = delta
    return g, delta