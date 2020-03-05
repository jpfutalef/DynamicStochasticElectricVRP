import numpy as np
import networkx as nx
from typing import List
from res.Node import CustomerNode, ChargeStationNode, DepotNode, NetworkNode
import matplotlib.pyplot as plt


class Network(nx.DiGraph):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.coordinates = {}
        self.ids_depot = []
        self.ids_customer = []
        self.ids_charge_stations = []

    def set_nodes(self, nodes):
        for node in nodes:
            self.add_node(node.id, attr=node)
            self.coordinates[node.id] = node.pos

            if node.isDepot():
                self.ids_depot.append(node.id)
            elif node.isCustomer():
                self.ids_customer.append(node.id)
            else:
                self.ids_charge_stations.append(node.id)

    def set_travel_time(self, travel_time: dict):
        for node_from, node_to_dict in travel_time.items():
            for node_to, tt in node_to_dict.items():
                self.add_edge(node_from, node_to, travel_time=tt)

    def set_energy_consumption(self, energy_consumption: dict):
        for node_from, node_to_dict in energy_consumption.items():
            for node_to, ec in node_to_dict.items():
                self.add_edge(node_from, node_to, energy_consumption=ec)

    def t(self, node_from, node_to, time_of_day=0.0):
        tt = self[node_from][node_to]['travel_time']
        return tt

    def e(self, node_from, node_to, payload, time_of_day=0.0):
        ec = self[node_from][node_to]['energy_consumption']
        return ec

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
        color = [color['attr'].color for _, color in self.nodes.items()]
        nx.draw_networkx(self, with_labels=True, pos=self.coordinates, node_color=color)
        plt.show()
