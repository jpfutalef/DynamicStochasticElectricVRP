import res.models.Fleet as Fleet
from typing import Union
import networkx as nx
import matplotlib.pyplot as plt

Network = Fleet.Network

NetworkType = Union[Network.Network, Network.DeterministicCapacitatedNetwork, Network.CapacitatedGaussianNetwork]
FleetType = Union[Fleet.Fleet, Fleet.GaussianFleet]


def draw_network(network: NetworkType, color=('lightskyblue', 'limegreen', 'goldenrod'), shape=('s', 'o', '^'),
                 fig: plt.Figure = None, save_to=None, **kwargs):
    nodes = network.nodes.keys()
    nodes_id = {i: i for i, node in network.nodes.items()}
    arcs = [(i, j) for i in nodes for j in nodes if i != j]

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(arcs)

    if not fig:
        fig = plt.figure()

    pos = {i: (network.nodes[i].pos_x, network.nodes[i].pos_y) for i in nodes}
    nx.draw(g, pos=pos, nodelist=network.depots, arrows=False, node_color=color[0], node_shape=shape[0],
            labels=nodes_id, **kwargs)
    nx.draw(g, pos=pos, nodelist=network.customers, arrows=False, node_color=color[1], node_shape=shape[1],
            labels=nodes_id, **kwargs)
    nx.draw(g, pos=pos, nodelist=network.charging_stations, arrows=False, node_color=color[2], node_shape=shape[2],
            labels=nodes_id, **kwargs)
    if save_to:
        fig.savefig(save_to)
    return fig, g


def to_networkx(network: NetworkType):
    nodes = network.nodes.keys()
    arcs = [(i, j) for i in nodes for j in nodes if i != j]
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(arcs)
    return g
