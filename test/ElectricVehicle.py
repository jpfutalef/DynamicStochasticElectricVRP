from dataclasses import dataclass
from typing import Union, Tuple, Dict

import numpy as np
from numpy import ndarray, array

from models.Edge import Edge
from models.Network import Network, DynamicNetwork
from models.Node import DepotNode, CustomerNode, ChargeStationNode
from models.ElectricVehicle import ElectricVehicle

if __name__ == '__main__':
    # Nodes
    depot = DepotNode(0)
    customer1 = CustomerNode(1, 15, .4)
    customer2 = CustomerNode(2, 15, .4)
    charge_station1 = ChargeStationNode(3)

    nodes = {0: depot, 1: customer1, 2: customer2, 3: charge_station1}

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
    n = Network(nodes, edges)

    # Now the EV
    ev = ElectricVehicle(0)
    ev.set_customers_to_visit((1, 2))
    route_without_cs = ((0, 1, 2, 0), (0, 0, 0, 0))
    route_with_cs = ((0, 1, 3, 2, 0), (0, 0, 10.5, 0, 0))

    # Route with CS
    x1_0 = 10 * 60
    x2_0 = 80
    x3_0 = sum([n.nodes[i].requiredDemand() for i in ev.assigned_customers])
    ev.set_route(route_without_cs, 10 * 60, 80, x3_0)
    ev.iterate_space(n)

    a = 1
