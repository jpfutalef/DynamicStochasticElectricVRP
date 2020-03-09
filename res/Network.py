import numpy as np
import networkx as nx
from typing import List
from res.Node import CustomerNode, ChargeStationNode, DepotNode, NetworkNode
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


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

    # Import tools
    def from_xml(self, path):
        # Open XML file
        tree = ET.parse(path)
        _info = tree.find('info')
        _network = tree.find('network')
        _fleet = tree.find('fleet')

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

        # Instance Network
        self.set_nodes(nodes)
        self.set_travel_time(travel_time)
        self.set_energy_consumption(energy_consumption)
        return
