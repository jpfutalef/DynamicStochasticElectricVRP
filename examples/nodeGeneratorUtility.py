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


# %% md User defined sizes for the whole network and amounts of customer, CSs and depots

depotAmount = 1
chargeStationsAmount = 2
customersAmount = 27

path = '../data/GA_implementation/'

networkSize = depotAmount + chargeStationsAmount + customersAmount

doAllWork = True
createTimeMatrix = True
createEnergyMatrix = True
createInfoFile = True

if createTimeMatrix or doAllWork:
    lowVal = 8.0
    upVal = 20.0
    matrix = res.EV_utilities.randomMatrixWithZeroDiagonal(networkSize, lower=lowVal, upper=upVal)
    name = 'timeMatrix_' + str(networkSize) + 'nodes.csv'
    df = pd.DataFrame(data=matrix)
    df.index.name = 'TT(MIN)'
    df.to_csv(path + name)


if createEnergyMatrix or doAllWork:
    lowVal = .5
    upVal = 5.
    matrix = res.EV_utilities.randomMatrixWithZeroDiagonal(networkSize, lower=lowVal, upper=upVal)
    name = 'energyMatrix_' + str(networkSize) + 'nodes.csv'
    df = pd.DataFrame(data=matrix)
    df.index.name = 'ENERGY(AH)'
    df.to_csv(path + name)

if createInfoFile or doAllWork:
    name = 'infoMatrix_' + str(networkSize) + 'nodes.csv'

    idNodes = range(depotAmount, networkSize)

    depotNodes = [x for x in range(0, depotAmount)]
    customerNodes = random.sample(idNodes, customersAmount)
    csNodes = [x for x in idNodes if x not in customerNodes]

    info = {}

    for i in depotNodes:
        info[i] = {'TYPE': ['DEPOT'], 'POS_X': [0.0], 'POS_Y': [0.0]}

    for i in customerNodes:
        serviceTime = random.uniform(1.0, 10.0)
        twLow = random.uniform(60 * 8.0, 60 * 18.0)
        twUp = twLow + serviceTime + random.uniform(30.0, 60 * 2.0)
        info[i] = {'TYPE': ['CUSTOMER'], 'POS_X': [np.random.uniform(-150.0, 150.0)],
                   'POS_Y': [np.random.uniform(-150.0, 150.0)],
                   'DEMAND': random.uniform(0.05, 0.5),
                   'SERVICE_TIME': serviceTime,
                   'TIME_WINDOW_LOW': twLow,
                   'TIME_WINDOW_UP': twUp}

    for i in csNodes:
        info[i] = {'TYPE': ['CS'], 'POS_X': [np.random.uniform(-150.0, 150.0)],
                   'POS_Y': [np.random.uniform(-150.0, 150.0)]}

    df = res.EV_utilities.makeInfoMatrix(info)
    print(df)
    print('Saving to:', path + name)
    df.to_csv(path+name)

    print('Number of depots:', len(depotNodes))
    print('Number of customers:', len(customerNodes))
    print('Number of CSs:', len(csNodes))

