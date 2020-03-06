# %%

# Too work with arguments and script paths
import sys

# scientific libraries and utilities
import numpy as np
import pandas as pd
import random
import time
import copy

# GA library
from deap import base
from deap import creator
from deap import tools

# Visualization tools
import matplotlib.pyplot as plt

# To reload modules
import importlib

# XML tools
import xml.etree.ElementTree as ET

# EV and network libraries
import res.GA_prev_test
from res.ElectricVehicle import ElectricVehicle
from res.Node import DepotNode, CustomerNode, ChargeStationNode
from res.Network import Network

t0 = time.time()

sys.path.append('..')

# %%
# 1. Specify instance name
instanceName = 'd1c7cs2_ev2'

# 2. Specify file and folder paths
folderPath = '../data/GA_implementation_xml/' + instanceName + '/'
filePath = folderPath + instanceName + '.xml'
print('Opening:', filePath)

# 3. Open XML file
tree = ET.parse(filePath)
_info = tree.find('info')
_network = tree.find('network')
_fleet = tree.find('fleet')

# %%
# 4. Store data
# [START Node data]
_nodes = _network.find('nodes')
_edges = _network.find('edges')
_technologies = _network.find('technologies')

nodes = []
for _node in _nodes:
    node_type = _node.get('type')
    id_node = int(_node.get('id'))
    pos = (float(_node.get('cx')), float(_node.get('cy')))
    if node_type == '0':
        node = DepotNode(id_node, pos=pos)

    elif node_type == '1':
        tw_upp = float(_node.get('tw_upp'))
        tw_low = float(_node.get('tw_low'))
        demand = float(_node.get('request'))
        spent_time = float(_node.get('spent_time'))
        node = CustomerNode(id_node, spent_time, demand, tw_upp, tw_low, pos=pos)

    elif node_type == '2':
        index_technology = int(_node.get('technology')) - 1
        _technology = _technologies[index_technology]
        charging_times = []
        battery_levels = []
        for _bp in _technology:
            charging_times.append(float(_bp.get('charging_time')))
            battery_levels.append(float(_bp.get('battery_level')))
        capacity = _node.get('capacity')
        node = ChargeStationNode(id_node, capacity, charging_times, battery_levels, pos=pos)

    nodes.append(node)

networkSize = len(nodes)

print('There are', networkSize, 'nodes in the network.')

# [END Node data]
# %%
# [START Edge data]
id_nodes = [x.id for x in nodes]
travel_time = {}
energy_consumption = {}
coordinates = {}

for i, nodeFrom in enumerate(_edges):
    tt_dict = travel_time[i] = {}
    ec_dict = energy_consumption[i] = {}
    for j, nodeTo in enumerate(nodeFrom):
        tt_dict[j] = float(nodeTo.get('travel_time'))
        ec_dict[j] = float(nodeTo.get('energy_consumption'))
    coordinates[i] = nodes[i].pos

# Show stored values
print('NODES IDSs:\n', id_nodes)
print('RESULTING TIME MATRIX:\n', travel_time)
print('RESULTING ENERGY CONSUMPTION MATRIX:\n', energy_consumption)
print('RESULTING NODES COORDINATES:\n', coordinates)
# [END Edge data]

# %%
# 5. Instance Network
net = Network()
net.set_nodes(nodes)
net.set_travel_time(travel_time)
net.set_energy_consumption(energy_consumption)
#net.draw()

# Usage example
f = 5
t = 6
key = 'energy_consumption'
print(key, net[f][t][key])
# %%
# 6. EVs attributes
numVehicles = int(_fleet.find('fleet_size').text)

attrib = {}
for _attrib in _fleet.find('vehicle_attributes'):
    attrib[_attrib.tag] = float(_attrib.text)

print('EV attributes:', attrib, '\n')

# 7. Proposal of how many customers each EV will visit
customers_per_car = [int(len(net.ids_customer) / numVehicles)] * numVehicles
if len(net.ids_customer) % numVehicles != 0:
    customers_per_car[-1] = int(len(net.ids_customer) / numVehicles) + 1

for i, j in enumerate(customers_per_car):
    print('Car', i, 'must visit', j, 'customer/s')
print('\n')

# %%
# 8. Instantiate EVS with initial sequences and attributes

vehicles = {}

ids_customer = [[1, 2, 3, 4], [5, 6, 7]]
for id_car, customers in enumerate(ids_customer):
    # sequences
    print('Car', id_car, 'must visit customers with ID:', customers)

    # Other variables
    soc_init = 80

    # instantiate
    ev = vehicles[id_car] = ElectricVehicle(id_car, net, **attrib)
    ev.set_customers_to_visit(customers)

# %%
# 9. Initialize GA
attrib_ga = {'allowed_charging_operations': 2, 'vehicles': vehicles}
i0, i1, i2 = res.GA_prev_test.createImportantIndices(**attrib_ga)
indices = [i0, i1, i2]
soc_init = 80.0

penalization_constant = -500000
weights = (1.0, 1.0, 1.0, 1.0)

# Build toolbox and useful classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Individual initializer
toolbox.register("individual", res.GA_prev_test.createRandomIndividual, **attrib_ga)

# Fitness, crossover, mutation and selection
toolbox.register("evaluate", res.GA_prev_test.fitness_raw, weights=weights, penalization_constant=penalization_constant,
                 x2_0=soc_init, **attrib_ga)
toolbox.register("mate", res.GA_prev_test.crossover, indices=indices, index=None, **attrib_ga)
toolbox.register("mutate", res.GA_prev_test.mutate, indices=indices, index=None, **attrib_ga )
toolbox.register("select", tools.selTournament, tournsize=3)

# Useful to decode
toolbox.register("decode", res.GA_prev_test.decode, **attrib_ga)

# %% the algorithm

tInitGA = time.time()
# Population TODO create function
n = 100
generations = 170

pop = []
for i in range(0, n):
    pop.append(creator.Individual(toolbox.individual()))

# CXPB  is the probability with which two individuals
#       are crossed
#
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.4, 0.4

# Evaluate the entire population
# fitnesses = list(map(toolbox.evaluate, pop))

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

bestOfAll = tools.selBest(pop, 1)[0]

print("################  Start of evolution  ################")
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

    Ymax.append(-max(fits))
    Ymin.append(-min(fits))
    Yavg.append(mean)
    Ystd.append(std)

    bestInd = tools.selBest(pop, 1)[0]
    print("Best individual: ", bestInd)

    worstInd = tools.selWorst(pop, 1)[0]
    print("Worst individual: ", worstInd)

    # Save best ind
    if bestInd.fitness.values[0] > bestOfAll.fitness.values[0]:
        bestOfAll = bestInd

# %%
print("################  End of (successful) evolution  ################")