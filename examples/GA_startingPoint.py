# %% md
# Simple E-VRP implementation

import importlib

import sys

import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from deap.tools import cxOnePoint
from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

import res.EV_utilities
import res.GA_utilities_1

t0 = time.time()

networkSize = 10

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

vehiclesDict = {}

customersID = [[2, 3, 4, 5], [6, 7, 9]]
nCustomers = sum([len(x) for x in customersID])

for carId, customersToVisit in enumerate(customersID):
    print('Car', carId, 'must visit customers with ID:', customersToVisit)

    # IMPORTANT: the proposed nodeSequence
    nodeSequence = [0] + customersToVisit + [0]
    chargingSequence = [0] * len(nodeSequence)

    # instantiate
    Qi = 80.0
    sumDi = np.sum([networkDict[i].demand for i in nodeSequence])
    vehiclesDict[carId] = res.EV_utilities.ElectricVehicle(carId, customersToVisit, networkDict,
                                                           nodeSequence=nodeSequence, chargingSequence=chargingSequence,
                                                           timeMatrix=timeMatrix.iat, energyMatrix=energyMatrix.iat,
                                                           x2=Qi, x3=sumDi)

for i in vehiclesDict.keys():
    seq = []
    for nodeId in vehiclesDict[i].nodeSequence:
        nodeStr = str(nodeId)
        seq.append(str(nodeStr + '  (' + networkDict[nodeId].getTypeAbbreviation() + ')'))
    print('The nodes sequence for vehicle', i, 'is', seq)
    print('The charging sequence for vehicle', i, 'is', vehiclesDict[i].chargingSequence, '\n')

# %% Genetic algorithm
print("##### GA #####")
importlib.reload(res.GA_utilities_1)

# allowed charging operations
numChargeOp = 2

# an example individual
ind1 = [2, 3, 4, 5, 3, 8, 10.0, -1, 4, 2, 720., 6, 7, 9, -1, 1, 10.0, -1, 1, 10., 740.]
ind2 = [5, 4, 3, 2, -1, 8, 10.0, 3, 4, 2, 750., 9, 7, 6, -1, 1, 10.0, -1, 1, 10., 760.]

# their fitness
fit1 = res.GA_utilities_1.fitness(ind1, vehiclesDict, allowed_charging_operations=numChargeOp)
fit2 = res.GA_utilities_1.fitness(ind2, vehiclesDict, allowed_charging_operations=numChargeOp)
print("fitness 1: ", fit1)
print("fitness 2: ", fit2)

# %% A crossover operation
print("ind 1: ", ind1)
print("ind 2: ", ind2)

print("*** crossover ***")
ind1New, ind2New = res.GA_utilities_1.crossover(ind1, ind2, vehiclesDict, allowed_charging_operations=numChargeOp)
print("ind 1 new: ", ind1New)
print("ind 2 new: ", ind2New, "\n")

# %% 4 mutation operations
print("*** mutation ***")
importlib.reload(res.GA_utilities_1)
ind1Mut1 = res.GA_utilities_1.mutate(ind1, vehiclesDict)
ind1Mut2 = res.GA_utilities_1.mutate(ind1, vehiclesDict, index=1)
ind1Mut3 = res.GA_utilities_1.mutate(ind1, vehiclesDict, index=5)
ind1Mut4 = res.GA_utilities_1.mutate(ind1, vehiclesDict, index=10)

# %% A random individual
importlib.reload(res.GA_utilities_1)
randomIndividual = res.GA_utilities_1.createRandomIndividual(vehiclesDict)

print("A random generated individual: ", randomIndividual, "\n")

# %% Using DEAP

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Individual initializer
toolbox.register("individual", res.GA_utilities_1.createRandomIndividual, vehiclesDict,
                 allowed_charging_operations=numChargeOp)

# Fitness, crossover, mutation and selection
toolbox.register("evaluate", res.GA_utilities_1.fitness, vehiclesDict=vehiclesDict)
toolbox.register("mate", res.GA_utilities_1.crossover, vehiclesDict=vehiclesDict)
toolbox.register("mutate", res.GA_utilities_1.mutate, vehiclesDict=vehiclesDict)
toolbox.register("select", tools.selTournament, tournsize=3)

# Useful to decode
toolbox.register("decode", res.GA_utilities_1.decodeFunction, vehiclesDict=vehiclesDict)

# Constraint handling
toolbox.register("distance", res.GA_utilities_1.distanceToFeasibleZone, vehicleDict=vehiclesDict)
toolbox.register("feasible", res.GA_utilities_1.feasibleIndividual, vehicleDict=vehiclesDict)
toolbox.decorate("evaluate", tools.DeltaPenality(toolbox.feasible, -5000.0, toolbox.distance))

# %% the algorithm
# Population TODO create function
n = 250
generations = 100

pop = []
for i in range(0, n):
    pop.append(creator.Individual(toolbox.individual()))

# CXPB  is the probability with which two individuals
#       are crossed
#
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.5, 0.2

print("Start of evolution")

# Evaluate the entire population
# fitnesses = list(map(toolbox.evaluate, pop)) FIXME por que esto entrega tuplas algunas veces y otras no?

for ind in pop:
    fit = toolbox.evaluate(ind)
    ind.fitness.values = fit

print("  Evaluated %i individuals" % len(pop))

# Extracting all the fitnesses of
fits = [ind.fitness.values[0] for ind in pop]

# Variable keeping track of the number of generations
g = 0
Ymax = []
Ymin = []
Yavg = []
Ystd = []
X = []

# Begin the evolution
while g < generations:
    # A new generation
    g = g + 1
    X.append(g)
    print("-- Generation %i --" % g)

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):

        # cross two individuals with probability CXPB
        if random.random() < CXPB:
            toolbox.mate(child1, child2)

            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:

        # mutate an individual with probability MUTPB
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    for ind in invalid_ind:
        fit = toolbox.evaluate(ind)
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(invalid_ind))

    # The population is entirely replaced by the offspring
    pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)

    Ymax.append(max(fits))
    Ymin.append(min(fits))
    Yavg.append(mean)
    Ystd.append(std)

    bestInd = tools.selBest(pop, 1)[0]
    print("Best individual: ", bestInd)

    worstInd = tools.selWorst(pop, 1)[0]
    print("Worst individual: ", worstInd)

print("-- End of (successful) evolution --")

bestInd = tools.selBest(pop, 1)[0]
print("Best individual: ", bestInd)
print("Fitness of best (stored): ", bestInd.fitness.values[0])
print("Fitness of best (calculated): ", toolbox.evaluate(bestInd))

worstInd = tools.selWorst(pop, 1)[0]
print("Worst individual: ", worstInd)
print("Fitness of best (stored): ", worstInd.fitness.values[0])
print("Fitness of best (calculated): ", toolbox.evaluate(worstInd))

# %%
fig1, ax1 = plt.subplots(1)
n = 2000
plt.plot(X[0:n], Ymax[0:n], '*', alpha=1)
plt.plot(X[0:n], Ymin[0:n], '*', alpha=.5)
plt.legend(('Best fitness', 'Worst fitness'))
plt.xlabel('Generations', fontsize=14)
plt.ylabel('Fitness', fontsize=14)
plt.title('Best vs worst fitness per generation (constrained problem)', fontsize=14)
# plt.xlim((-200, 2100))
# plt.ylim((-130, -60))

ax1.arrow(2180, -125, 0., 50, width=1., head_width=20, head_length=5, clip_on=False)
th1 = plt.text(2300, -105, 'better', fontsize=14,
               rotation=90, rotation_mode='anchor')

plt.show()

fig2, ax2 = plt.subplots(1)
n = 2000
plt.plot(X[0:n], Ystd[0:n], '*', alpha=1)
plt.xlabel('Generations', fontsize=14)
plt.ylabel('Standar deviation', fontsize=14)
plt.title('Std per generation', fontsize=14)

plt.show()

# %% sequences visualization of best
toolbox.evaluate(bestInd)

stateSequences = {}

for vehicleId in vehiclesDict.keys():
    seqEta = np.zeros(vehiclesDict[vehicleId].si)
    stateSequences[vehicleId] = vehiclesDict[vehicleId].createStateSequenceStatic(seqEta)

# Vehicle 1

nSeq = vehiclesDict[0].nodeSequence
kCustomers = []
tWindowsCenter = []
tWindowsWidth = []
for i, node in enumerate(nSeq):
    if networkDict[node].isCustomer():
        kCustomers.append(i)
        tWindowsCenter.append((networkDict[node].timeWindowUp + networkDict[node].timeWindowDown)/2.0)
        tWindowsWidth.append(networkDict[node].timeWindowUp - networkDict[node].timeWindowDown)

plt.subplot(231)
plt.plot(stateSequences[0][0, :], '-o', markersize=3, linewidth=1)
plt.errorbar(kCustomers, tWindowsCenter, yerr=tWindowsWidth, fmt=',', capsize=2)
plt.title('Reaching/leaving times')
plt.xlabel('k')
plt.ylabel('X1')

plt.subplot(232)
plt.plot(80*np.ones_like(stateSequences[0][1, :]), '--k')
plt.plot(40*np.ones_like(stateSequences[0][1, :]), '--k')
plt.plot(stateSequences[0][1, :], '-o', markersize=3, linewidth=1)
plt.title('SOC at each stop')
plt.xlabel('k')
plt.ylabel('X2')

plt.subplot(233)
plt.plot(stateSequences[0][2, :], '-o', markersize=3, linewidth=1)
plt.title('Payload')
plt.xlabel('k')
plt.ylabel('X3')

# Vehicle 2

nSeq = vehiclesDict[1].nodeSequence
kCustomers = []
tWindowsCenter = []
tWindowsWidth = []
for i, node in enumerate(nSeq):
    if networkDict[node].isCustomer():
        kCustomers.append(i)
        tWindowsCenter.append((networkDict[node].timeWindowUp + networkDict[node].timeWindowDown)/2.0)
        tWindowsWidth.append(networkDict[node].timeWindowUp - networkDict[node].timeWindowDown)

plt.subplot(234)
plt.plot(stateSequences[1][0, :], '-o', markersize=3, linewidth=1)
plt.errorbar(kCustomers, tWindowsCenter, yerr=tWindowsWidth, fmt=',', capsize=2)
plt.title('Reaching/leaving times')
plt.xlabel('k')
plt.ylabel('X1')

plt.subplot(235)
plt.plot(80*np.ones_like(stateSequences[1][1, :]), '--k')
plt.plot(40*np.ones_like(stateSequences[1][1, :]), '--k')
plt.plot(stateSequences[1][1, :], '-o', markersize=3, linewidth=1)
plt.title('SOC at each stop')
plt.xlabel('k')
plt.ylabel('X2')

plt.subplot(236)
plt.plot(stateSequences[1][2, :], '-o', markersize=3, linewidth=1)
plt.title('Payload')
plt.xlabel('k')
plt.ylabel('X3')

plt.show()
plt.close()