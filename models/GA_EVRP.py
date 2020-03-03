# %%

# Too work with arguments and script paths
import sys

# scientific libraries and utilities
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

# GA library
from deap import base
from deap import creator
from deap import tools

# Visualization tools
from bokeh.plotting import figure, output_notebook, show
from bokeh.layouts import gridplot, column
from bokeh.layouts import row as layout_row
from bokeh.models.annotations import Arrow, Label
from bokeh.models.arrow_heads import OpenHead, NormalHead, VeeHead
from bokeh.models import ColumnDataSource, Div, Whisker, Span, Range1d
from bokeh.io import export_svgs, export_png

# To reload modules
import importlib

# XML tools
import xml.etree.ElementTree as ET

# Copy tools
import copy

# Display useful loading of bokeh library and start timer
output_notebook()
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
from res.Node import DepotNode, CustomerNode, ChargeStationNode

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
timeMatrix = np.zeros((networkSize, networkSize))
energyMatrix = np.zeros((networkSize, networkSize))
coordinates = {}

for i, nodeFrom in enumerate(_edges):
    for j, nodeTo in enumerate(nodeFrom):
        timeMatrix[i][j] = float(nodeTo.get('travel_time'))
        energyMatrix[i][j] = float(nodeTo.get('energy_consumption'))
    coordinates[i] = nodes[i].pos

# Show stored values
print('NODES IDSs:\n', id_nodes)
print('RESULTING TIME MATRIX:\n', timeMatrix)
print('RESULTING ENERGY CONSUMPTION MATRIX:\n', energyMatrix)
print('RESULTING NODES COORDINATES:\n', coordinates)
# [END Edge data]

# %%
# 5. Instance Network
from res.Network import Network

net = Network(nodes, timeMatrix, energyMatrix)
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
from res.EV import ElectricVehicle

vehicles_dict = {}
ids_customer = copy.deepcopy(net.ids_customer)
for id_car, num_customers in enumerate(customers_per_car):
    # sequences
    ids_customer_to_visit = []
    for j in range(0, num_customers):
        index = random.randint(0, len(ids_customer) - 1)
        ids_customer_to_visit.append(ids_customer.pop(index))
    print('Car', id_car, 'must visit customers with ID:', ids_customer_to_visit)

    # First decision variables
    node_sequence = [0] + ids_customer_to_visit + [0]
    charging_sequence = [0] * len(node_sequence)
    departure_time = 24.0 * 30.0

    # Other variables
    attrib['x2_0'] = 80

    # instantiate
    vehicles_dict[id_car] = ev = ElectricVehicle(id_car, node_sequence, charging_sequence, departure_time, net, **attrib)
    r, l = ev.iterateState()
    print('node seq:', ev.node_sequence)
    print('spent times:', (l-r)[0,:])
    print('spent times:', ev.get_spent_times())
    print('travel times', [r_next-l_prev for r_next, l_prev in zip(r[0,1:], l[0,0:-1])] )
    print('travel times',  ev.get_travel_times())
# %%


