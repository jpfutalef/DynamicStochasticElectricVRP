# Utility TODO documentation

import sys

import numpy as np
import pandas as pd
import random


# %% Useful functions
def randomMatrixWithZeroDiagonal(size, lower=5.0, upper=15.0, save=True, file_name='randomMatrix',
                                 folder_path='../data/', indexName='VAL (UNIT)'):
    # TODO add doc
    randomMatrix = np.random.uniform(lower, upper, size=(size, size))
    np.fill_diagonal(randomMatrix, 0.0)
    print('Resulting matrix with name', file_name, ':\n', randomMatrix)
    if save:
        filePath = folder_path + file_name + '.csv'
        print('Saving to ', filePath)
        data_frame = pd.DataFrame(data=randomMatrix)
        data_frame.index.name = indexName
        data_frame.to_csv(filePath)
    return randomMatrix


def makeInfoMatrix(nodesInfo: dict, save=True, file_name='infoMatrix', folder_path='../data/'):
    # TODO add doc
    data_frame = pd.DataFrame()
    for nodeID, nodeInfo in nodesInfo.items():
        row = pd.DataFrame(nodeInfo, index=(nodeID,))
        row.index.name = 'ID'
        data_frame = data_frame.append(row)
    print('Obtained info matrix:\n', data_frame)
    if save:
        filePath = folder_path + file_name + '.csv'
        print('Saving to ', filePath)
        data_frame.to_csv(filePath)
    return data_frame


# %% md User defined sizes for the whole network and amounts of customer, CSs and depots

depotAmount = 1
customersAmount = 25
chargeStationsAmount = 2

if len(sys.argv) > 1:
    depotAmount = int(sys.argv[1])
    customersAmount = int(sys.argv[2])
    chargeStationsAmount = int(sys.argv[3])

folderPath = '../data/GA_implementation/'

networkSize = depotAmount + chargeStationsAmount + customersAmount

createTimeMatrix = True
createEnergyMatrix = True
createInfoFile = True

if createTimeMatrix:
    lowVal = 8.0
    upVal = 20.0
    fileName = 'timeMatrix_' + str(networkSize) + 'nodes'
    timeMatrix = randomMatrixWithZeroDiagonal(networkSize, lower=lowVal, upper=upVal, file_name=fileName,
                                              folder_path=folderPath, indexName='TT(MIN)')

if createEnergyMatrix:
    lowVal = .5
    upVal = 5.
    fileName = 'energyMatrix_' + str(networkSize) + 'nodes'
    energyMatrix = randomMatrixWithZeroDiagonal(networkSize, lower=lowVal, upper=upVal, file_name=fileName,
                                                folder_path=folderPath, indexName='ENERGY(AH)')

if createInfoFile:
    fileName = 'infoMatrix_' + str(networkSize) + 'nodes'

    depotNodes = [x for x in range(0, depotAmount)]
    customerNodes = [x for x in range(depotAmount, depotAmount + customersAmount)]
    csNodes = [x for x in range(depotAmount+customersAmount, networkSize)]

    info = {}

    for i in depotNodes:
        info[i] = {'TYPE': ['DEPOT'],
                   'POS_X': [0.0],
                   'POS_Y': [0.0]}

    for i in customerNodes:
        # Service times bounds in minutes
        serviceTime_lower = 1.0
        serviceTime_upper = 10.0

        # Time window bounds in minutes
        timeWindow_lower_lower = 60.*10.
        timeWindow_lower_upper = 60.*14.
        timeWindow_upper_lower = 60.
        timeWindow_upper_upper = 60.*3.

        # Create times
        serviceTime = random.uniform(serviceTime_lower, serviceTime_upper)
        twLow = random.uniform(timeWindow_lower_lower, timeWindow_lower_upper)
        twUp = twLow + serviceTime + random.uniform(timeWindow_upper_lower, timeWindow_upper_upper)

        # Save to dictionary
        info[i] = {'TYPE': ['CUSTOMER'],
                   'POS_X': [np.random.uniform(-150.0, 150.0)],
                   'POS_Y': [np.random.uniform(-150.0, 150.0)],
                   'DEMAND': random.uniform(0.05, 0.5),
                   'SERVICE_TIME': serviceTime,
                   'TIME_WINDOW_LOW': twLow,
                   'TIME_WINDOW_UP': twUp}

    for i in csNodes:
        info[i] = {'TYPE': ['CS'],
                   'POS_X': [np.random.uniform(-150.0, 150.0)],
                   'POS_Y': [np.random.uniform(-150.0, 150.0)]}

    dataFrame = makeInfoMatrix(info, file_name=fileName, folder_path=folderPath)

    print('Number of depots:', len(depotNodes))
    print('Number of customers:', len(customerNodes))
    print('Number of CSs:', len(csNodes))
