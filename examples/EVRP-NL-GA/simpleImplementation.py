# %% md
# Simple E-VRP implementation

import importlib

import sys

import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

import res.EV_utilities
import res.GA_utilities_1

t0 = time.time()

networkSize = 40

# %% md
# choose if do random matrices
doAllWork = True
createTimeMatrix = True
createEnergyMatrix = True
createInfoFile = True

if createTimeMatrix and doAllWork:
    lowVal = 4.0
    upVal = 15.0
    res.EV_utilities.randomMatrixWithZeroDiagonal(networkSize, lower=lowVal, upper=upVal,
                                                  name='timeMatrixRandom',
                                                  indexName='TT(MIN)')
if createEnergyMatrix and doAllWork:
    lowVal = 0.1
    upVal = 10.0
    res.EV_utilities.randomMatrixWithZeroDiagonal(networkSize, lower=lowVal, upper=upVal,
                                                  name='energyMatrixRandom',
                                                  indexName='ENERGY(AH)')

if createInfoFile and doAllWork:
    depotNodes = [0]
    idNodes = range(1, networkSize)
    #customerNodes = random.sample(idNodes, random.randint(1, networkSize - 1))
    customerNodes = random.sample(idNodes, 37)
    csNodes = [x for x in idNodes if x not in customerNodes]

    if len(csNodes) > len(customerNodes):
        a = customerNodes
        customerNodes = csNodes
        csNodes = a

    info = {}

    for i in depotNodes:
        info[i] = {'TYPE': ['DEPOT'], 'POS_X': [0.0], 'POS_Y': [0.0]}

    for i in customerNodes:
        serviceTime = random.uniform(1.0, 10.0)
        twLow = random.uniform(60 * 8.0, 60 * 18.0)
        twUp = twLow + serviceTime + random.uniform(0, 60 * 2.0)
        info[i] = {'TYPE': ['CUSTOMER'], 'POS_X': [np.random.uniform(-150.0, 150.0)],
                   'POS_Y': [np.random.uniform(-150.0, 150.0)],
                   'DEMAND': random.uniform(0.05, 0.5),
                   'SERVICE_TIME': serviceTime,
                   'TIME_WINDOW_LOW': twLow,
                   'TIME_WINDOW_UP': twUp}

    for i in csNodes:
        info[i] = {'TYPE': ['CS'], 'POS_X': [np.random.uniform(-150.0, 150.0)],
                   'POS_Y': [np.random.uniform(-150.0, 150.0)]}

    res.EV_utilities.saveInfoMatrix(info, name='infoMatrix')

    print('Number of depots:', len(depotNodes))
    print('Number of customers_per_vehicle:', len(customerNodes))
    print('Number of CSs:', len(csNodes))

# %% md
# Import time and energy matrices and show a value example

pathTM = "../data/simpleImplementation/timeMatrixRandom" + str(networkSize) + ".csv"
timeMatrix = pd.read_csv(pathTM).set_index("TT(MIN)")

print(timeMatrix, "\n")

pathEM = '../data/simpleImplementation/energyMatrixRandom' + str(networkSize) + '.csv'
energyMatrix = pd.read_csv(pathEM).set_index("ENERGY(AH)")

print(energyMatrix, '\n')

# Indexing example
t = timeMatrix.iat[1, 2]
print("Travel time from 1 to 2: ", t)

e = energyMatrix.iat[1, 2]
print("Energy consumption from 1 to 2: ", e)

# %% md
# Import nodes information to differentiate among them

path = '../data/simpleImplementation/timeMatrixRandom_info' + str(networkSize) + '.csv'
infoMatrix = pd.read_csv(path)

depotDF = infoMatrix[infoMatrix['TYPE'] == 'DEPOT'].dropna(axis=1)
customerDF = infoMatrix[infoMatrix['TYPE'] == 'CUSTOMER'].dropna(axis=1)
csDF = infoMatrix[infoMatrix['TYPE'] == 'CS'].dropna(axis=1)

print('Depots DF:\n', depotDF, '\n')
print('Customers DF:\n', customerDF, '\n')
print('Charge Stations DF\n', csDF, '\n')

# %% md

# Create list with information

networkDict = {'DEPOT_LIST': [], 'CUSTOMER_LIST': [], 'CS_LIST': [],
               'TIME_MATRIX': timeMatrix, 'ENERGY_MATRIX': energyMatrix}

for _, row in depotDF.iterrows():
    networkDict[row['ID']] = res.EV_utilities.DepotNode(row['ID'])
    networkDict['DEPOT_LIST'].append(networkDict[row['ID']])

for _, row in customerDF.iterrows():
    networkDict[row['ID']] = res.EV_utilities.CustomerNode(row['ID'], row['SERVICE_TIME'], row['DEMAND'],
                                                           timeWindowUp=row['TIME_WINDOW_UP'],
                                                           timeWindowDown=row['TIME_WINDOW_LOW'])
    networkDict['CUSTOMER_LIST'].append(networkDict[row['ID']])

constantChargingTime = False

for _, row in csDF.iterrows():
    # Simple CS with linear curve
    if constantChargingTime:
        networkDict[row['ID']] = res.EV_utilities.ChargeStationNode(row['ID'])
    else:
        networkDict[row['ID']] = res.EV_utilities.ChargeStationNode(row['ID'],
                                                                    timePoints=[0.0, 240.0], socPoints=[0.0, 100.0])
    networkDict['CS_LIST'].append(networkDict[row['ID']])

# Information

# %% md

# Number of cars and their random set to visit. Ensure that each one of them visit at least one

t1 = time.time()

nVehicles = 2
insertChargeStations = True
doRandomInsertion = False
randomChargeLow = 5.0
randomChargeUpp = 10.0

vehiclesDict = {}

customersId = [evID for evID in customerDF['ID']]
nCustomers = len(customersId)
nCustomersPerCar = [int(nCustomers / nVehicles)] * nVehicles

if len(customersId) % nVehicles != 0:
    nCustomersPerCar[-1] = int(len(customersId) / nVehicles) + 1

for i, j in enumerate(nCustomersPerCar):
    print('Car', i, 'must visit', j, 'customer/s')
print('\n')

for carId, nCustomersCar in enumerate(nCustomersPerCar):
    carCustomersId = []
    for j in range(0, nCustomersCar):
        index = random.randint(0, len(customersId) - 1)
        carCustomersId.append(customersId.pop(index))
    customersToVisit = [customerId for customerId in carCustomersId]
    print('Car', carId, 'must visit customers_per_vehicle with ID:', customersToVisit)

    # IMPORTANT: the proposed nodeSequence
    nodeSequence = [0] + customersToVisit + [0]
    chargingSequence = [0] * len(nodeSequence)
    print('Car ', carId, 'will visit customers_per_vehicle in the following order:', nodeSequence, '\n')
    if insertChargeStations and len(nodeSequence) > 3:
        for j, nodeId in enumerate(nodeSequence[2:-1]):
            if random.randint(0, 1):
                csId = random.sample([x.id for x in networkDict['CS_LIST']], 1)[0]
                nodeSequence.insert(j + 2, csId)
                chargingSequence.insert(j + 2, np.random.uniform(randomChargeLow, randomChargeUpp))
    # instantiate
    Qi = 100.0
    sumDi = np.sum([networkDict[i].demand for i in nodeSequence])
    vehiclesDict[carId] = res.EV_utilities.ElectricVehicle(carId, customersToVisit, networkDict,
                                                           nodeSequence=nodeSequence, chargingSequence=chargingSequence,
                                                           timeMatrix=timeMatrix.iat, energyMatrix=energyMatrix.iat,
                                                           x2=Qi, x3=sumDi)

# Print the final nodeSequence
if insertChargeStations:
    print("Charge stations have been inserted...")
else:
    print("No charge stations inserted")

for i in vehiclesDict.keys():
    seq = []
    for nodeId in vehiclesDict[i].nodeSequence:
        nodeStr = str(nodeId)
        seq.append(str(nodeStr + '  (' + networkDict[nodeId].getTypeAbbreviation() + ')'))
    print('The nodes sequence for vehicle', i, 'is', seq)
    print('The charging sequence for vehicle', i, 'is', vehiclesDict[i].chargingSequence, '\n')

# %% md

# Amount of rows each constraint will occupy at the A matrix
sumSi = np.sum([len(vehiclesDict[vehicleId].nodeSequence) for vehicleId in vehiclesDict])
lenK0 = 2 * np.sum([len(vehiclesDict[vehicleId].nodeSequence) - 1 for vehicleId in vehiclesDict]) + 1

constraintRows = 0

# 16
constraintRows += nVehicles

# 17
constraintRows += nCustomers

# 18
constraintRows += nCustomers

# 24
constraintRows += lenK0 * len(networkDict['CS_LIST'])

# 25.1
constraintRows += sumSi

# 25.2
constraintRows += sumSi

# 26.1
constraintRows += np.sum(list(vehiclesDict.keys()))

# 26.2
constraintRows += np.sum(list(vehiclesDict.keys())) + 2  # FIXME de donde viene este +2?

# %% md

# Create linear inequalities matrix Ax <= b
sizeOfOpVector = (5 + 2 * networkSize) * sumSi - 2 * len(vehiclesDict) * networkSize + networkSize

A = np.zeros((constraintRows, sizeOfOpVector))
b = np.zeros((constraintRows, 1))

rowToChange = 0

# 16
for i, j in enumerate(vehiclesDict):
    i1 = 2 * sumSi + i * vehiclesDict[j].si
    i2 = 2 * sumSi + (i + 1) * vehiclesDict[j].si - 1

    A[rowToChange, i1] = -1.0
    A[rowToChange, i2] = 1.0
    b[rowToChange] = vehiclesDict[j].maxTourDuration
    rowToChange += 1
    # print('16:', rowToChange)

# 24
for k0 in range(lenK0):
    for cs in networkDict['CS_LIST']:
        i1 = 5 * sumSi + cs.id + k0 * networkSize
        A[rowToChange, i1] = 1.0
        b[rowToChange] = networkDict[cs.id].maximumParallelOperations
        rowToChange += 1
        # print('24:', rowToChange)

# 25.1 & 25.2
for i, j in enumerate(vehiclesDict):
    for k in range(vehiclesDict[j].si):
        i1 = 4 * sumSi + + i * vehiclesDict[j].si + k

        A[rowToChange, i1] = -1.0
        b[rowToChange] = vehiclesDict[j].alphaDown
        rowToChange += 1
        # print('25.1:', rowToChange)

        A[rowToChange, i1] = 1.0
        b[rowToChange] = vehiclesDict[j].alphaUp
        rowToChange += 1
        # print('25.2:', rowToChange)

# 26.1 & 26.2
for i, j in enumerate(vehiclesDict):
    i1 = 4 * sumSi + (i + 1) * vehiclesDict[j].si - 1

    A[rowToChange, i1] = -1.0
    b[rowToChange] = vehiclesDict[j].betaDown
    rowToChange += 1
    # print('26.1:', rowToChange)

    A[rowToChange, i1] = 1.0
    b[rowToChange] = vehiclesDict[j].betaUp
    rowToChange += 1
    # print('26.2:', rowToChange)

# %% md

# Linear inequalities with conditioned sets

# 17 & 18
for i, j in enumerate(vehiclesDict):
    for cId in vehiclesDict[j].customersId:
        k = vehiclesDict[j].nodeSequence.index(cId)
        i1 = 2 * sumSi + k

        A[rowToChange, i1] = -1.0
        b[rowToChange] = -networkDict[cId].timeWindowDown
        rowToChange += 1
        # print('17:', rowToChange)

        A[rowToChange, i1] = 1.0
        b[rowToChange] = networkDict[cId].timeWindowUp
        rowToChange += 1
        # print('18:', rowToChange)

# %% md

# Equality constraints for state

stateSequences = {}

for vehicleId in vehiclesDict.keys():
    seqEta = np.zeros(vehiclesDict[vehicleId].si)
    stateSequences[vehicleId] = vehiclesDict[vehicleId].createStateSequenceStatic(seqEta)

# %% md

# Counting vector dynamics

leavingReachingSequences = {}

for vehicleId in vehiclesDict.keys():
    seqEta = np.zeros(vehiclesDict[vehicleId].si)
    leavingReachingSequences[vehicleId] = vehiclesDict[vehicleId].createReachingLeavingTimes(seqEta)

theta = np.zeros(networkSize)
theta[0] = nVehicles

aX = []
for vehicleId in leavingReachingSequences.keys():
    x = []
    for k, nodeId in enumerate(vehiclesDict[vehicleId].nodeSequence):
        # agregar ts, nodo, tipo, vehiculo
        if k == 0:
            x.append([leavingReachingSequences[vehicleId][1][k], nodeId, -1, vehicleId])
        elif k == len(vehiclesDict[vehicleId].nodeSequence) - 1:
            x.append([leavingReachingSequences[vehicleId][1][k], nodeId, 1, vehicleId])
        else:
            x.append([leavingReachingSequences[vehicleId][0][k], nodeId, 1, vehicleId])
            x.append([leavingReachingSequences[vehicleId][1][k], nodeId, -1, vehicleId])
    aX.append(x)

# Ordenar
oX = []
for i in range(0, lenK0 - 1):  # FIXME por qué -1?
    c = float("inf")
    v = 0
    for j in range(0, nVehicles):
        if not aX[j]:
            pass
        elif aX[j][0][0] < c:
            c = aX[j][0][0]
            v = j
    popEvent = aX[v].pop(0)  # notice that aX will be empty after this. rerun to obtain new results
    oX.append(popEvent)

# print(oX)

thetaMatrix = np.zeros((networkSize, lenK0))
thetaMatrix[:, 0] = theta

# FIXME arreglar todo esto
for i in range(0, len(oX)):
    g, typ = res.EV_utilities.gamma(oX, i, networkSize)
    # print('theta(',i,')\n', thetaMatrix[:, i])
    # print('gamma(',i,')\n', g)
    thetaMatrix[:, i + 1] = thetaMatrix[:, i] + g

# %% md

# Cost function of sequences

travelTimeCost = np.sum([vehiclesDict[x].travelTimeCost for x in vehiclesDict.keys()])
chargingTimeCost = np.sum([vehiclesDict[x].chargingTimeCost for x in vehiclesDict.keys()])

print('Total travel time:', travelTimeCost, 'min')
print('Total charging time:', chargingTimeCost, 'min')
print('Total charging cost:')
print('Total cost:', travelTimeCost + chargingTimeCost, '\n')

# %% md

# Performance

tEnd = time.time()

print('Initial time:', t0)
print('Ending time:', tEnd)
print('Delta time global:', tEnd - t0)
print('Delta time matrices:', tEnd - t1)

# %% md
plt.subplot(131)
plt.plot(stateSequences[0][0, :], '-o')
plt.title('Accumulated time')
plt.xlabel('k')
plt.ylabel('X1')

plt.subplot(132)
plt.plot(stateSequences[0][1, :], '-o')
plt.title('SOC at each stop')
plt.xlabel('k')
plt.ylabel('X2')

plt.subplot(133)
plt.plot(stateSequences[0][2, :], '-o')
plt.title('Payload')
plt.xlabel('k')
plt.ylabel('X3')

# plt.show()

# %% Genetic algorithm
importlib.reload(res.GA_utilities_1)

# allowed charging operations
numChargeOp = 2

# an example individual
ind = [4, 2, 6, 2, 8, 10.0, -1, 4, 2, 720.,
       5, 7, 9, 3, 7, 1, 10.0, -1, 1, 10., 740.]

# its fitness
S, L, x0 = res.GA_utilities_1.decodeFunction(ind, vehiclesDict, allowed_charging_operations=numChargeOp)
