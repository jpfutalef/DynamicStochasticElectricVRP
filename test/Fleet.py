from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

from models.ElectricVehicle import ElectricVehicle
from models.Network import Network
from models.Node import DepotNode, CustomerNode, ChargeStationNode
from models.Edge import Edge
from models.Fleet import Fleet

import numpy as np
from numpy import ndarray, zeros, array
import xml.etree.ElementTree as ET


if __name__ == '__main__':
    # Nodes
    depot = DepotNode(0)
    customer1 = CustomerNode(1, 15, .4)
    customer2 = CustomerNode(2, 15, .4)
    customer3 = CustomerNode(3, 15, .4)
    customer4 = CustomerNode(4, 15, .4)
    charge_station1 = ChargeStationNode(5)

    nodes = {0: depot, 1: customer1, 2: customer2, 3: customer3, 4: customer4, 5: charge_station1}

    # Edges
    edges = {}
    for node_from in nodes.keys():
        n_from = edges[node_from] = {}
        for node_to in nodes.keys():
            if node_from == node_to:
                tt = 0
                ec = 0
            else:
                tt = 15.0
                ec = 18.0
            n_from[node_to] = Edge(node_from, node_to, tt, ec)

    # Create a static network
    print('***** STATIC NETWORK *****')
    network = Network(nodes, edges)

    # Now the EVs
    ev1 = ElectricVehicle(1)
    ev2 = ElectricVehicle(2)

    ev1.set_customers_to_visit((1, 2))
    ev2.set_customers_to_visit((3, 4))

    # The fleet
    fleet = Fleet(network, {1: ev1, 2: ev2}, (1, 2))

    # Routes and initial conditions
    x1_0_1 = 10 * 60
    x2_0_1 = 80
    x3_0_1 = sum([network.nodes[i].requiredDemand() for i in ev1.assigned_customers])

    x1_0_2 = 8 * 60
    x2_0_2 = 80
    x3_0_2 = sum([network.nodes[i].requiredDemand() for i in ev2.assigned_customers])

    route1 = ((0, 1, 2, 0), (0, 0, 0, 0))
    route2 = ((0, 3, 5, 4, 0), (0, 0, 10.5, 0, 0))

    routes = {1: (route1, x1_0_1, x2_0_1, x3_0_1), 2: (route2, x1_0_2, x2_0_2, x3_0_2)}

    # Set routes and iterate
    fleet.set_routes_of_vehicles(routes)

    # Get optimization vector
    op_vector = fleet.create_optimization_vector()

    a = 1
