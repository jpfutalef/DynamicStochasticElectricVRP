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
import res
from res.ElectricVehicle import ElectricVehicle, feasible, createOptimizationVector
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
# net.draw()

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

vehicles_dict = {}
ids_customer = [[1, 2, 3, 4], [5, 6, 7]]
for id_car, customers in enumerate(ids_customer):
    # sequences
    print('Car', id_car, 'must visit customers with ID:', customers)

    # First decision variables
    node_sequence = [0] + customers + [0]
    charging_sequence = [0] * len(node_sequence)
    departure_time = 24.0 * 30.0

    # Other variables
    soc_init = 80

    # instantiate
    ev = vehicles_dict[id_car] = ElectricVehicle(id_car, net, **attrib)
    ev.set_customers_to_visit(customers)
    ev.set_sequences(node_sequence, charging_sequence, departure_time, soc_init)
    r, l = ev.iterateState()
    print('node seq:', ev.node_sequence)
    print('spent times:', (l - r)[0, :])
    print('spent times:', ev.get_spent_times())
    print('travel times', [r_next - l_prev for r_next, l_prev in zip(r[0, 1:], l[0, 0:-1])])
    print('travel times', ev.get_travel_times())
# %%

# Test feasibility functions
x = createOptimizationVector(vehicles_dict)
print(x)
print(feasible(x, vehicles_dict))

# make an over charge
seqs_node = [[0, 1, 2, 8, 3, 4, 0], [0, 5, 6, 7, 9, 0]]
seqs_charging = [[0, 0, 0, 80., 0, 0, 0], [0, 0, 0, 0, 90., 0]]
for id_ev, vehicle in vehicles_dict.items():
    vehicle.set_sequences(seqs_node[id_ev], seqs_charging[id_ev], 800, 100)
    r, l = vehicle.iterateState()

x = createOptimizationVector(vehicles_dict)
print(x)
print(feasible(x, vehicles_dict))

