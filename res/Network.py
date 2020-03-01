import numpy as np
import networkx as nx
from typing import List
from res.Node import CustomerNode, ChargeStationNode, DepotNode


class Network(nx.DiGraph):
    def __init__(self, nodes: List[CustomerNode, ChargeStationNode, DepotNode],
                 travel_time=None, energy_consumption=None, **attr):
        super().__init__(**attr)
        for node in nodes:
            self.add_node(node.id, attr=node)

        if travel_time:
            self.travel_time_worker(travel_time)

        if energy_consumption:
            self.energy_consumption_worker(energy_consumption)

    def travel_time_worker(self, travel_time):
        for node0 in travel_time:
            for node1 in node0:
                self.add_edge(node0, node1, travel_time=travel_time[node0][node1])

    def energy_consumption_worker(self, energy_consumption):
        for node0 in energy_consumption:
            for node1 in node0:
                self.add_edge(node0, node1, energy_consumption=energy_consumption[node0][node1])

    def travel_time(self, node_from, node_to, time_of_day=0.0):
        return self[node_from][node_to]['travel_time']

    def energy_consumption(self, node_from, node_to, time_of_day=0.0):
        return self[node_from][node_to]['energy_consumption']

    def spent_time(self, node, p=0, q=0):
        return self.nodes[node]['attr'].spentTime(p, q)

    def required_demand(self, node):
        return self.nodes[node]['attr'].requiredDemand()
