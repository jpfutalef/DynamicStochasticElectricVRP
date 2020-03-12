import numpy as np
import pandas as pd
import networkx as nx

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

    def isCustomer(self):  # TODO add doc
        return False

    def getTypeAbbreviation(self):  # TODO add doc
        return 'NN'


class ChargeStationNode(NetworkNode):
    # TODO add documentation
    # TODO unit test
    def __init__(self, nodeId, maximumParallelOperations=4, timePoints=(0.0, 120.0), socPoints=(0.0, 100.0)):
        super().__init__(nodeId)
        self.maximumParallelOperations = maximumParallelOperations
        self.timePoints = timePoints
        self.socPoints = socPoints

    def calculateTimeSpent(self, initSOC, endSOC):
        # FIXME actually, verify it is working properly
        # TODO verify complexity
        # TODO add documentation
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

    def isCustomer(self):
        return True


class ElectricVehicle:  # TODO add documentation
    networkInfo: list
    customersToVisit: np.ndarray
    id: int
    timeMatrix: list

    def __init__(self, evId, customersToVisitId, networkInfo, nodeSequence=None, chargingSequence=None,
                 maxPayload=2.0, batteryCapacity=200.0, maxTourDuration=300.0, alphaDown=40.0, alphaUp=80.0,
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

        # the number of customers_per_vehicle to visit
        self.customerCount = len(customersToVisitId)

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
        self.x2_leaving = x2
        self.x3_leaving = x3
        self.state = np.array([x1, x2, x3])

        # Internal costs
        self.travelTimeCost = 0.0
        self.energyConsumptionCost = 0.0
        self.chargingTimeCost = 0.0
        self.chargingCost = 0.0

    def F1(self, idK0: int, idK1: int, Lk, eta):
        # TODO incorporate time travel function
        spentTime = self.networkInfo[idK0].spentTime(self.x2, Lk)
        travelTime = self.timeMatrix[idK0, idK1]
        x1LeavingCurrent = self.x1 + spentTime
        x1ReachingNext = x1LeavingCurrent + travelTime
        return x1ReachingNext, x1LeavingCurrent

    def F2(self, idK0, idK1, Lk, eta):
        energyConsumption = self.energyMatrix[idK0, idK1]
        x2ReachingNext = self.x2 + Lk - energyConsumption
        x2LeavingCurrent = self.x2 + Lk
        return x2ReachingNext, x2LeavingCurrent  # TODO incorporate energy consumption

    def F3(self, idK0, idK1, Lk, eta):
        x3ReachingNext = self.x3 - self.networkInfo[idK0].requiredDemand()
        x3LeavingCurrent = self.x3 - self.networkInfo[idK0].requiredDemand()
        return x3ReachingNext, x3LeavingCurrent

    def indexInSequence(self, nodeId):
        return

    def stateUpdate(self, idK0, idK1, Lk, eta):
        self.x1, self.x1_leaving = self.F1(idK0, idK1, Lk, eta)
        self.x2, self.x2_leaving = self.F2(idK0, idK1, Lk, eta)
        self.x3, self.x3_leaving = self.F3(idK0, idK1, Lk, eta)
        self.state = np.array([self.x1, self.x2, self.x3])

    def returnToInitialCondition(self):
        self.x1 = self.x1_0
        self.x2 = self.x2_0
        self.x3 = self.x3_0
        self.x1_leaving = self.x1_0
        self.x2_leaving = self.x2_0
        self.x3_leaving = self.x3_0
        self.state = np.array([self.x1, self.x2, self.x3])

    def createStateSequenceStatic(self, sequenceEta):  # TODO docu
        # FIXME nodeSequence nodes and nodeSequence recharge should be pass to instantiation
        # TODO notice thar this function starts from the initial state
        self.returnToInitialCondition()
        X = np.zeros((3, len(self.nodeSequence)))
        for k, nodeId in enumerate(self.nodeSequence):
            X[:, k] = self.state
            if k == len(self.nodeSequence) - 1:
                pass
            else:
                # TODO incorporate the following to functions
                self.travelTimeCost += self.timeMatrix[self.nodeSequence[k], self.nodeSequence[k + 1]]
                self.energyConsumptionCost += self.energyMatrix[self.nodeSequence[k], self.nodeSequence[k + 1]]
                if self.networkInfo[self.nodeSequence[k]].isChargeStation():
                    self.chargingTimeCost += self.networkInfo[self.nodeSequence[k]].spentTime(self.x3,
                                                                                              self.chargingSequence[k])
                self.stateUpdate(self.nodeSequence[k], self.nodeSequence[k + 1], self.chargingSequence[k],
                                 sequenceEta[k])
        self.returnToInitialCondition()
        return X

    def createReachingLeavingStates(self, sequenceEta):  # TODO docu
        X = np.zeros((6, len(self.nodeSequence)))
        for k, nodeId in enumerate(self.nodeSequence):
            X[0, k] = self.x1
            X[2, k] = self.x2
            X[4, k] = self.x3
            if k == len(self.nodeSequence) - 1:
                X[1, k] = self.x1
                X[3, k] = self.x2
                X[5, k] = self.x3
            else:
                self.stateUpdate(self.nodeSequence[k], self.nodeSequence[k + 1], self.chargingSequence[k],
                                 sequenceEta[k])
                X[1, k] = self.x1_leaving
                X[3, k] = self.x2_leaving
                X[5, k] = self.x3_leaving
        self.returnToInitialCondition()
        return X

    def updateSequences(self, nodeSequence, chargingSequence, x1):
        self.nodeSequence = nodeSequence
        self.chargingSequence = chargingSequence
        self.x1_0 = x1
        try:
            self.si = len(nodeSequence)
        except TypeError:
            self.si = 0


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


def createOptimizationVector(nodeSequences, chargeSequences, x0Sequence, vehiclesDict):
    # FIXME do not create state sequences here. Or maybe?
    # FIXME preallocate to arrays before to make more efficient
    S = []
    L = []
    X1 = []
    X2 = []
    X3 = []

    for _, nodeSeq in nodeSequences.items():
        S += nodeSeq

    for _, chargeSeq in chargeSequences.items():
        L += chargeSeq

    # ensure reset and iterate
    for vehicleID, v in vehiclesDict.items():
        v.travelTimeCost = 0
        v.chargingTimeCost = 0
        v.energyConsumptionCost = 0

        v.updateSequences(nodeSequences[vehicleID], chargeSequences[vehicleID], x0Sequence[vehicleID])
        seqEta = np.zeros(v.si)
        X = v.createStateSequenceStatic(seqEta)
        X1 += list(X[0, :])
        X2 += list(X[1, :])
        X3 += list(X[2, :])

        v.travelTimeCost = 0
        v.chargingTimeCost = 0
        v.energyConsumptionCost = 0

    # TODO does this affect integer nature of sequences?
    V = np.vstack(S + L + X1 + X2 + X3)
    return V


def feasible(nodeSequences, chargeSequences, x0Sequence, vehiclesDict):
    # Obtain optimization vector
    X = createOptimizationVector(nodeSequences, chargeSequences, x0Sequence, vehiclesDict)
    # print("nodeSequences", nodeSequences)
    # print("chargeSequences", chargeSequences)
    # print("x0Sequence", x0Sequence)
    # LINEAR CONTRAINTS
    # Amount of vehicles and customers_per_vehicle, and the network information dictionary
    nVehicles = len(vehiclesDict)
    nCustomers = np.sum([x.customerCount for _, x in vehiclesDict.items()])
    networkDict = vehiclesDict[0].networkInfo

    networkSize = len(networkDict['DEPOT_LIST']) + len(networkDict['CUSTOMER_LIST']) + len(networkDict['CS_LIST'])

    # Amount of rows each constraint will occupy at the A matrix
    sumSi = np.sum([len(x) for _, x in nodeSequences.items()])

    constraintRows = 0

    # 16
    constraintRows += nVehicles

    # 17
    constraintRows += nCustomers

    # 18
    constraintRows += nCustomers

    # 25.1
    constraintRows += sumSi

    # 25.2
    constraintRows += sumSi

    # 26.1
    constraintRows += sumSi

    # 26.2
    constraintRows += sumSi

    # K0
    lenK0 = 2 * sumSi - 2 * nVehicles

    # Create linear inequalities matrix Ax <= b
    sizeOfOpVector = 5 * sumSi

    A = np.zeros((constraintRows, sizeOfOpVector))
    b = np.zeros((constraintRows, 1))

    # Start filling
    rowToChange = 0

    # 16
    i1 = 2 * sumSi
    i2 = 2 * sumSi
    for j in vehiclesDict:
        i2 += len(nodeSequences[j]) - 1

        A[rowToChange, i1] = -1.0
        A[rowToChange, i2] = 1.0
        b[rowToChange] = vehiclesDict[j].maxTourDuration
        rowToChange += 1
        # print('16:', rowToChange)
        i1 += len(nodeSequences[j])
        i2 += 1

    # 17 & 18
    i1 = 2 * sumSi
    for j in vehiclesDict:
        for nodeID in nodeSequences[j]:
            if networkDict[nodeID].isCustomer():
                A[rowToChange, i1] = -1.0
                b[rowToChange] = -networkDict[nodeID].timeWindowDown
                rowToChange += 1
                # print('17:', rowToChange)

                A[rowToChange, i1] = 1.0
                b[rowToChange] = networkDict[nodeID].timeWindowUp - networkDict[nodeID].serviceTime
                rowToChange += 1
                # print('18:', rowToChange)
            i1 += 1

    # 25.1 & 25.2
    i1 = 3 * sumSi
    for i, j in enumerate(vehiclesDict):
        for k in range(len(nodeSequences[j])):
            A[rowToChange, i1] = -1.0
            b[rowToChange] = -vehiclesDict[j].alphaDown
            rowToChange += 1
            # print('25.1:', rowToChange)

            A[rowToChange, i1] = 1.0
            b[rowToChange] = vehiclesDict[j].alphaUp
            rowToChange += 1
            # print('25.2:', rowToChange)
            i1 += 1

    # 26.1 & 26.2
    i1 = 3 * sumSi
    for i, j in enumerate(vehiclesDict):
        for k in range(len(nodeSequences[j])):
            A[rowToChange, i1] = -1.0
            b[rowToChange] = -vehiclesDict[j].alphaDown + vehiclesDict[j].chargingSequence[k]
            rowToChange += 1
            # print('25.1:', rowToChange)

            A[rowToChange, i1] = 1.0
            b[rowToChange] = vehiclesDict[j].alphaUp - vehiclesDict[j].chargingSequence[k]
            rowToChange += 1
            # print('25.2:', rowToChange)
            i1 += 1

    # Check
    mult = np.matmul(A, X)
    for res, cond in zip(mult, b):
        if not res[0] <= cond[0]:
            return False
    return True


def distance(nodeSequences, chargeSequences, x0Sequence, vehiclesDict, allowed_charging_operations=2):
    # Obtain optimization vector
    X = createOptimizationVector(nodeSequences, chargeSequences, x0Sequence, vehiclesDict)
    # print("nodeSequences", nodeSequences)
    # print("chargeSequences", chargeSequences)
    # print("x0Sequence", x0Sequence)
    # LINEAR CONTRAINTS
    # Amount of vehicles and customers_per_vehicle, and the network information dictionary
    nVehicles = len(vehiclesDict)
    nCustomers = np.sum([x.customerCount for _, x in vehiclesDict.items()])
    networkDict = vehiclesDict[0].networkInfo

    networkSize = len(networkDict['DEPOT_LIST']) + len(networkDict['CUSTOMER_LIST']) + len(networkDict['CS_LIST'])

    # Amount of rows each constraint will occupy at the A matrix
    sumSi = np.sum([len(x) for _, x in nodeSequences.items()])

    constraintRows = 0

    # 16
    constraintRows += nVehicles

    # 17
    constraintRows += nCustomers

    # 18
    constraintRows += nCustomers

    # 25.1
    constraintRows += sumSi

    # 25.2
    constraintRows += sumSi

    # 26.1
    constraintRows += sumSi

    # 26.2
    constraintRows += sumSi

    # Create linear inequalities matrix Ax <= b
    sizeOfOpVector = 5 * sumSi

    A = np.zeros((constraintRows, sizeOfOpVector))
    b = np.zeros((constraintRows, 1))

    # Start filling
    rowToChange = 0

    # 16
    i1 = 2 * sumSi
    i2 = 2 * sumSi
    for j in vehiclesDict:
        i2 += len(nodeSequences[j]) - 1

        A[rowToChange, i1] = -1.0
        A[rowToChange, i2] = 1.0
        b[rowToChange] = vehiclesDict[j].maxTourDuration
        rowToChange += 1
        # print('16:', rowToChange)
        i1 += len(nodeSequences[j])
        i2 += 1

    # 17 & 18
    i1 = 2 * sumSi
    for j in vehiclesDict:
        for nodeID in nodeSequences[j]:
            if networkDict[nodeID].isCustomer():
                A[rowToChange, i1] = -1.0
                b[rowToChange] = -networkDict[nodeID].timeWindowDown
                rowToChange += 1
                # print('17:', rowToChange)

                A[rowToChange, i1] = 1.0
                b[rowToChange] = networkDict[nodeID].timeWindowUp - networkDict[nodeID].serviceTime
                rowToChange += 1
                # print('18:', rowToChange)
            i1 += 1

    # 25.1 & 25.2
    i1 = 3 * sumSi
    for i, j in enumerate(vehiclesDict):
        for k in range(len(nodeSequences[j])):
            A[rowToChange, i1] = -1.0
            b[rowToChange] = -vehiclesDict[j].alphaDown
            rowToChange += 1
            # print('25.1:', rowToChange)

            A[rowToChange, i1] = 1.0
            b[rowToChange] = vehiclesDict[j].alphaUp
            rowToChange += 1
            # print('25.2:', rowToChange)
            i1 += 1

    # 26.1 & 26.2
    i1 = 3 * sumSi
    for i, j in enumerate(vehiclesDict):
        i1 += len(nodeSequences[j]) - 1

        A[rowToChange, i1] = -1.0
        b[rowToChange] = vehiclesDict[j].betaDown
        rowToChange += 1
        # print('26.1:', rowToChange)

        A[rowToChange, i1] = 1.0
        b[rowToChange] = vehiclesDict[j].betaUp
        rowToChange += 1
        # print('26.2:', rowToChange)

        i1 += 1

    # Check
    mult = np.matmul(A, X)
    boolList = [0] * np.size(b)

    for i, (res, cond) in enumerate(zip(mult, b)):
        if not res[0] <= cond[0]:
            boolList[i] = False
        else:
            boolList[i] = True

    # Penalization per restriction
    dist = 0.0
    rowToCheck = 0

    # 16
    for j in vehiclesDict:
        if not boolList[rowToCheck]:
            # print("Unfeasible maximum service time.")
            dist += np.power(mult[rowToCheck, 0] - b[rowToCheck, 0], 2)
        rowToCheck += 1

    # 17
    for j in range(nCustomers):
        if not boolList[rowToCheck]:
            # print("Unfeasible TW lower bound.")
            dist += np.power(mult[rowToCheck, 0] - b[rowToCheck, 0], 2)
        rowToCheck += 1

    # 18
    for j in range(nCustomers):
        if not boolList[rowToCheck]:
            # print("Unfeasible TW upper bound.")
            dist += np.power(mult[rowToCheck, 0] - b[rowToCheck, 0], 2)
        rowToCheck += 1

    # 25.1
    for j in range(sumSi):
        if not boolList[rowToCheck]:
            # print("Unfeasible SOH pol1cy lower")
            dist += np.power(mult[rowToCheck, 0] - b[rowToCheck, 0], 2)
        rowToCheck += 1

    # 25.2
    for j in range(sumSi):
        if not boolList[rowToCheck]:
            # print("Unfeasible SOH policy upper")
            dist += np.power(mult[rowToCheck, 0] - b[rowToCheck, 0], 2)
        rowToCheck += 1

    # 26.1
    for j in range(sumSi):
        if not boolList[rowToCheck]:
            # print("Unfeasible SOH pol1cy lower")
            dist += np.power(mult[rowToCheck, 0] - b[rowToCheck, 0], 2)
        rowToCheck += 1

    # 26.2
    for j in range(sumSi):
        if not boolList[rowToCheck]:
            # print("Unfeasible SOH pol1cy lower")
            dist += np.power(mult[rowToCheck, 0] - b[rowToCheck, 0], 2)
        rowToCheck += 1

    return dist