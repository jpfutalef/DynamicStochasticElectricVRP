import numpy as np
import networkx as nx
from typing import List
from res.Node import CustomerNode, ChargeStationNode, DepotNode
import matplotlib.pyplot as plt


class Network(nx.DiGraph):
    def __init__(self, nodes, travel_time=None, energy_consumption=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.coordinates = []
        for node in nodes:
            self.add_node(node.id, attr=node)
            self.coordinates.append(node.pos)

        if travel_time is not None:
            self.travel_time_worker(travel_time)

        if energy_consumption is not None:
            self.energy_consumption_worker(energy_consumption)

    def travel_time_worker(self, travel_time: np.ndarray):
        shape = travel_time.shape
        for node0 in range(shape[0]):
            for node1 in range(shape[1]):
                self.add_edge(node0, node1, travel_time=travel_time[node0][node1])

    def energy_consumption_worker(self, energy_consumption):
        shape = energy_consumption.shape
        for node0 in range(shape[0]):
            for node1 in range(shape[1]):
                self.add_edge(node0, node1, energy_consumption=energy_consumption[node0][node1])

    def travel_time(self, node_from, node_to, time_of_day=0.0):
        return self[node_from][node_to]['travel_time']

    def energy_consumption(self, node_from, node_to, payload, time_of_day=0.0):
        return self[node_from][node_to]['energy_consumption']

    def spent_time(self, node, p=0, q=0):
        return self.nodes[node]['attr'].spentTime(p, q)

    def demand(self, node):
        return self.nodes[node]['attr'].requiredDemand()

    def getTypeAbbreviation(self, node):
        return self.nodes[node]['attr'].getTypeAbbreviation()

    def isDepot(self, node):
        return self.nodes[node]['attr'].isDepot()

    def isCustomer(self, node):
        return self.nodes[node]['attr'].isCustomer()

    def isChargeStation(self, node):
        return self.nodes[node]['attr'].isChargeStation()

    def draw(self):
        nx.draw_networkx(self, with_labels=True, pos=self.coordinates, node_color='orange')
        plt.show()
