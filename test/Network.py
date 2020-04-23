from typing import Tuple, Dict, Union

from models.Edge import Edge, DynamicEdge
from models.Node import CustomerNode, ChargeStationNode, DepotNode
from models.Network import Network, DynamicNetwork, from_xml, from_element_tree

from numpy import array


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

    i = 0
    print('Node %d is a' % n.nodes[i].id, n.nodes[i].getTypeAbbreviation())

    i = 0
    j = 1
    print("Travel time from {} to {} is {:.2f}".format(i, j, n.edges[i][j].get_travel_time()))

    # Create dynamic network
    print('***** DYNAMIC NETWORK *****')
    sampling_time = 15

    edges = {}
    for node_from in nodes.keys():
        n_from = edges[node_from] = {}
        for node_to in nodes.keys():
            if node_from == node_to:
                tt = array([0 for x in range(0, 24 * 60, sampling_time)])
                ec = array([0 for x in range(0, 24 * 60, sampling_time)])
            else:
                tt = array([15 for x in range(0, 8 * 60, sampling_time)] +
                           [25 for x in range(8 * 60, 11 * 60, sampling_time)] +
                           [15 for x in range(11 * 60, 18 * 60, sampling_time)] +
                           [25 for x in range(18 * 60, 21 * 60, sampling_time)] +
                           [15 for x in range(21 * 60, 24 * 60, sampling_time)])
                ec = array([12 for x in range(0, 8 * 60, sampling_time)] +
                           [22 for x in range(8 * 60, 11 * 60, sampling_time)] +
                           [12 for x in range(11 * 60, 18 * 60, sampling_time)] +
                           [22 for x in range(18 * 60, 21 * 60, sampling_time)] +
                           [12 for x in range(21 * 60, 24 * 60, sampling_time)])
            n_from[node_to] = DynamicEdge(node_from, node_to, tt, ec, sampling_time)

    n = DynamicNetwork(samp_time=sampling_time)
    n.set_nodes(nodes)
    n.set_edges(edges)

    i = 0
    print('Node %d is a' % n.nodes[i].id, n.nodes[i].getTypeAbbreviation())

    i = 0
    j = 1
    tod = 60 * 8
    print("Travel time from {} to {} at time {} is {:.2f}".format(i, j, tod, n.edges[i][j].get_travel_time(tod)))
    print("Travel time from {} to {} at time {} is {:.2f}".format(i, j, tod, n.t(i, j, tod)))

    # From file
    path = '../data/GA_implementation_xml/20C_4CS_1D_4EV/20C_4CS_1D_4EV.xml'
    network = from_xml(path)
    a=1

