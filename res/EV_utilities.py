import numpy as np
import pandas as pd

"""
Classes
"""


class NetworkNode:
    """
    A general network node which has an associated Id
    """

    def __init__(self, nodeId, spentTime=0, demand=0):
        """
        Simplest constructor with Id and a zero spent time
        :param nodeId: an int number with an associated Id
        """
        self.id = nodeId
        self.spentTimeAtNode = spentTime
        self.demand = demand

    def spentTime(self, p, q):
        """
        A function all subclasses must implement in order to know how much time it was spent at the node
        :param q: battery SOC of a vehicle reaching the node
        :param p: battery SOC increment of the vehicle leaving the node
        :return: the spent time
        """
        return self.spentTimeAtNode

    def requiredDemand(self):  # TODO add doc
        return self.demand

    def isChargeStation(self):  # TODO add doc
        return False

    def getTypeAbbreviation(self):  # TODO add doc
        return 'NN'


class ChargeStationNode(NetworkNode):
    # TODO add documentation
    # TODO unit test
    def __init__(self, nodeId, maximumParallelOperations=4, timeCalculationMethod='piecewise', timePoints=None,
                 socPoints=None):
        super().__init__(nodeId)
        self.maximumParallelOperations = maximumParallelOperations
        self.timeCalculationMethod = timeCalculationMethod
        if timeCalculationMethod == 'piecewise':
            self.timePoints = timePoints
            self.socPoints = socPoints

    def calculateTimeSpent(self, initSOC, endSOC):  # FIXME actually, verify it is working properly
        # TODO verify complexity
        # TODO add documentation
        if self.timePoints is None:
            return 6.0
        doInit = True
        doEnd = False
        initIndex = 0
        endIndex = 0
        for i, ai in enumerate(self.socPoints):
            if doInit and ai >= initSOC:
                initIndex = i - 1
                doInit = False
                doEnd = True
            if doEnd and ai >= endSOC:
                endIndex = i - 1
                break
        # init time
        m = (self.socPoints[initIndex + 1] - self.socPoints[initIndex]) / (self.timePoints[initIndex + 1] -
                                                                           self.timePoints[initIndex])
        n = self.socPoints[initIndex] - m * self.timePoints[initIndex]
        initTime = (initSOC - n) / m

        # end time
        m = (self.socPoints[endIndex + 1] - self.socPoints[endIndex]) / (self.timePoints[endIndex + 1] -
                                                                         self.timePoints[endIndex])
        n = self.socPoints[endIndex] - m * self.timePoints[endIndex]
        endTime = (endSOC - n) / m
        return endTime - initTime

    def spentTime(self, x3, L):
        return self.calculateTimeSpent(x3, x3 + L)

    def isChargeStation(self):
        return True

    def getTypeAbbreviation(self):  # TODO add doc
        return 'CS'


class DepotNode(NetworkNode):  # TODO add documentation
    def __init__(self, nodeId):
        super().__init__(nodeId)

    def getTypeAbbreviation(self):  # TODO add doc
        return 'DEPOT'


class CustomerNode(NetworkNode):  # TODO add documentation
    def __init__(self, nodeId, serviceTime, demand, timeWindowUp=None, timeWindowDown=None):
        super().__init__(nodeId, demand=demand)
        self.timeWindowDown = timeWindowDown
        self.timeWindowUp = timeWindowUp
        self.serviceTime = serviceTime

    def spentTime(self, p, q):
        return self.serviceTime

    def getTypeAbbreviation(self):  # FIXME maybe as an instance parameter?
        return 'C'


class ElectricVehicle:  # TODO add documentation
    networkInfo: list
    customersToVisit: np.ndarray
    id: int
    timeMatrix: list

    def __init__(self, evId, customersToVisitId, networkInfo, nodeSequence=None, chargingSequence=None,
                 maxPayload=2.0, batteryCapacity=40.0, maxTourDuration=300.0, alphaDown=40.0, alphaUp=80.0,
                 betaDown=45.0, betaUp=55.0, x1=0.0, x2=40.0, x3=2.0, timeMatrix=None, energyMatrix=None):
        """
        EV model

        :param evId: EV id
        :param customersToVisitId: array containing all customer Id that the EV must visit
        :param networkInfo: dictionary containing all nodes. Every node is positioned in the specified Id
        :param maxPayload: maximum allowed payload to be transported by the EV in tons
        :param batteryCapacity: battery capacity in Ah
        :param maxTourDuration: maximum tour time duration in minutes
        :param nodeSequence: a list with ordered node IDs representing the nodeSequence to follow the the EV
        :param chargingSequence: a list with ordered charging amounts representing the SOC increase at each node
        """

        self.id = evId
        self.customersId = customersToVisitId
        self.networkInfo = networkInfo

        self.maxPayload = maxPayload
        self.batteryCapacity = batteryCapacity
        self.maxTourDuration = maxTourDuration
        self.alphaDown = alphaDown
        self.alphaUp = alphaUp
        self.betaDown = betaDown
        self.betaUp = betaUp
        self.timeMatrix = timeMatrix
        self.energyMatrix = energyMatrix

        self.nodeSequence = nodeSequence
        self.chargingSequence = chargingSequence
        try:
            self.si = len(nodeSequence)
        except TypeError:
            self.si = 0

        # Save initial conditions
        self.x1_0 = x1
        self.x2_0 = x2
        self.x3_0 = x3
        self.state_0 = np.array([x1, x2, x3])

        # Internal state
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x1_leaving = x1
        self.state = np.array([x1, x2, x3])

        # Internal costs
        self.travelTimeCost = 0.0
        self.chargingTimeCost = 0.0
        self.chargingCost = 0.0

    def F1(self, idK0: int, idK1: int, Lk, eta):
        return self.x1 + self.networkInfo[idK0].spentTime(self.x2, Lk) + self.timeMatrix[
            idK0, idK1]  # TODO incorporate time travel function

    def F2(self, idK0, idK1, Lk, eta):
        return self.x2 + Lk - self.energyMatrix[idK0, idK1]  # TODO incorporate energy consumption

    def F3(self, idK0, idK1, Lk, eta):
        return self.x3 - self.networkInfo[idK0].requiredDemand()

    def indexInSequence(self, nodeId):
        return

    def stateUpdate(self, idK0, idK1, Lk, eta):
        self.x1 = self.F1(idK0, idK1, Lk, eta)
        self.x2 = self.F2(idK0, idK1, Lk, eta)
        self.x3 = self.F3(idK0, idK1, Lk, eta)
        self.state = np.array([self.x1, self.x2, self.x3])
        self.x1_leaving = self.x1 + self.networkInfo[idK1].spentTime(self.x2,
                                                                     Lk)  # TODO incorporate time travel function

    def returnToInitialCondition(self):
        self.x1 = self.x1_0
        self.x2 = self.x2_0
        self.x3 = self.x3_0
        self.x1_leaving = self.x1_0
        self.state = np.array([self.x1, self.x2, self.x3])

    def createStateSequenceStatic(self, sequenceEta):  # TODO docu
        # FIXME nodeSequence nodes and nodeSequence recharge should be pass to instantiation
        # TODO notice thar this function starts from the initial state
        X = np.zeros((3, len(self.nodeSequence)))
        for k, nodeId in enumerate(self.nodeSequence):
            X[:, k] = self.state
            if k == len(self.nodeSequence) - 1:
                pass
            else:
                # TODO incorporate the following to functions
                self.travelTimeCost += self.timeMatrix[self.nodeSequence[k], self.nodeSequence[k + 1]]
                if self.networkInfo[self.nodeSequence[k]].isChargeStation():
                    self.chargingTimeCost += self.networkInfo[self.nodeSequence[k]].spentTime(self.x3,
                                                                                              self.chargingSequence[k])
                self.stateUpdate(self.nodeSequence[k], self.nodeSequence[k + 1], self.chargingSequence[k],
                                 sequenceEta[k])
        self.returnToInitialCondition()
        return X

    def createReachingLeavingTimes(self, sequenceEta):  # TODO docu
        # FIXME nodeSequence nodes and nodeSequence recharge should be pass to instantiation
        # TODO notice thar this function starts from the initial state
        X = np.zeros((2, len(self.nodeSequence)))
        for k, nodeId in enumerate(self.nodeSequence):
            X[:, k] = np.asarray([self.x1, self.x1_leaving])
            if k == len(self.nodeSequence) - 1:
                pass
            else:
                # TODO incorporate the following to functions
                if self.networkInfo[self.nodeSequence[k]].isChargeStation():
                    self.chargingTimeCost += self.networkInfo[self.nodeSequence[k]].spentTime(self.x2,
                                                                                              self.chargingSequence[k])
                # FIXME why the charging sequence is k+1 in the following?
                self.stateUpdate(self.nodeSequence[k], self.nodeSequence[k + 1], self.chargingSequence[k+1],
                                 sequenceEta[k])
        self.returnToInitialCondition()
        return X


"""
Functions
"""


def gamma(tSeries, k0, nNodes):  # TODO add option to reuse a previous gamma
    """
    Calculates gamma vector and delta at instant k0

    :param tSeries: time series ordered in ascendant fashion according to global counter. Each element of tSeries has
                    the form [timeStamp, occurrenceNode, eventType, occurrenceVehicle]
    :param k0: instant to calculate gamma
    :param nNodes: amount of nodes in the network, including depot and CSs
    :return: tuple in the form (gammaVector, delta)
    """
    g = np.zeros(nNodes)
    occurrenceNode = tSeries[k0][1]
    delta = int(tSeries[k0][2])
    g[occurrenceNode] = delta
    return g, delta


def sumPathLengths(paths):
    """
    Calculates the sum of all given paths minus one. The paths are in the form [Path0, Path1, ..., Pathm]

    :param paths: an iterable containing all paths of the whole nodeSequence
    :return: a number with the sum of all path lengths minus one
    """
    K = 0
    for i in paths:
        K += len(i) - 1
    return K


def createGlobalCounterSet(sumOfPathLengths):
    """
    Creates the set of all possible values the global counter can be

    :param sumOfPathLengths: sum of all path lengths minus one
    :return: the set with the values for k0 counter
    """
    return range(0, 2 * sumOfPathLengths + 1)


def createEmptyInequalityMatrix(pathLengthsList):
    m: int = len(pathLengthsList)
    sumSi: int = int(np.sum(pathLengthsList))
    opVectorSize: int = 7 * sumSi - 2 * m + 1
    return


def randomMatrixWithZeroDiagonal(size, lower=5.0, upper=15.0, save=True, name='symmetricMatrix',
                                 indexName='Val'):  # TODO add doc
    timeMatrix = np.random.uniform(lower, upper, size=(size, size))
    np.fill_diagonal(timeMatrix, 0.0)
    if save:
        path = '../data/simpleImplementation/' + name + str(size) + '.csv'
        print('Saving to ', path)
        df = pd.DataFrame(data=timeMatrix)
        df.index.name = indexName
        df.to_csv(path)
    return timeMatrix


def saveInfoMatrix(nodeInfo: dict, name='timeMatrixRandom'):  # TODO add doc
    df = pd.DataFrame()
    for nodeID in nodeInfo.keys():
        row = pd.DataFrame(nodeInfo[nodeID], index=[nodeID])
        row.index.name = 'ID'
        df = df.append(row)
    print(df)
    path = '../data/simpleImplementation/' + name + '_info' + str(len(nodeInfo)) + '.csv'
    print('Saving to ', path)
    df.to_csv(path)
    return
