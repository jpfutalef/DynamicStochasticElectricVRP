from typing import Tuple, Dict, Union

from models.Edge import Edge, DynamicEdge
from models.Node import CustomerNode, ChargeStationNode, DepotNode
import xml.etree.ElementTree as ET
import xml.dom.minidom
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

NodeDict = Dict[int, Union[CustomerNode, ChargeStationNode, DepotNode]]
EdgeDict = Dict[int, Dict[int, Union[Edge, DynamicEdge]]]
DynamicEdgeDict = Dict[int, Dict[int, DynamicEdge]]
NodeType = Union[DepotNode, CustomerNode, ChargeStationNode]


class Network:
    nodes: NodeDict
    edges: EdgeDict
    depots: Tuple[int, ...]
    customers: Tuple[int, ...]
    charging_stations: Tuple[int, ...]

    def __init__(self, network_nodes: NodeDict = None, network_edges: EdgeDict = None):
        self.nodes = {}
        self.edges = {}

        self.depots = ()
        self.customers = ()
        self.charging_stations = ()

        if network_nodes:
            self.set_nodes(network_nodes)
        if network_edges:
            self.set_edges(network_edges)

    def __len__(self):
        return len(self.nodes)

    def set_nodes(self, node_collection: NodeDict):
        del self.nodes, self.depots, self.customers, self.charging_stations

        self.nodes = node_collection
        self.depots = ()
        self.customers = ()
        self.charging_stations = ()

        for node in node_collection.values():
            self.add_to_ids(node)

    def set_edges(self, edge_collection: EdgeDict):
        del self.edges
        self.edges = edge_collection

    def add_to_ids(self, node: NodeType):
        if node.isDepot():
            self.depots += (node.id,)
        elif node.isCustomer():
            self.customers += (node.id,)
        else:
            self.charging_stations += (node.id,)

    def add_node(self, node: NodeType):
        self.nodes[node.id] = node
        self.add_to_ids(node)

    def add_nodes(self, node_collection: NodeDict):
        for node in node_collection.values():
            self.add_node(node)

    def t(self, node_from: int, node_to: int, time_of_day: float) -> Union[float, int]:
        return self.edges[node_from][node_to].get_travel_time(time_of_day)

    def e(self, node_from: int, node_to: int, payload: float, vehicle_weight: float,
          time_of_day: float) -> Union[float, int]:
        return self.edges[node_from][node_to].get_energy_consumption(payload, vehicle_weight, time_of_day)

    def spent_time(self, node: int, p, q, eta=None):
        return self.nodes[node].spentTime(p, q, eta)

    def demand(self, node: int):
        return self.nodes[node].demand

    def isDepot(self, node: int):
        return self.nodes[node].isDepot()

    def isCustomer(self, node: int):
        return self.nodes[node].isCustomer()

    def isChargingStation(self, node: int):
        return self.nodes[node].isChargeStation()

    def xml_tree(self):
        _network = ET.Element('network')
        _nodes = ET.SubElement(_network, 'nodes')
        _edges = ET.SubElement(_network, 'edges')
        _technologies = ET.SubElement(_network, 'technologies')
        technologies = []

        for node in self.nodes.values():
            _nodes.append(node.xml_element())
            _node_from = ET.SubElement(_edges, 'node_from', attrib={'id': str(node.id)})
            for node_to in self.nodes.values():
                _node_to = self.edges[node.id][node_to.id].xml_element()
                _node_from.append(_node_to)
            if node.isChargeStation():
                if node.technology not in technologies:
                    _technology = ET.SubElement(_technologies, 'technology', attrib={'type': str(node.technology)})
                    for t, soc in zip(node.time_points, node.soc_points):
                        attrib = {'charging_time': str(t), 'battery_level': str(soc)}
                        _bp = ET.SubElement(_technology, 'breakpoint', attrib=attrib)
        return _network

    def write_xml(self, path, print_pretty=False):
        tree = self.xml_tree()
        if print_pretty:
            xml_pretty = xml.dom.minidom.parseString(ET.tostring(tree, 'utf-8')).toprettyxml()
            with open(path, 'w') as file:
                file.write(xml_pretty)
        else:
            ET.ElementTree(tree).write(path)

    def draw(self, color=('lightskyblue', 'limegreen', 'goldenrod'), shape=('s', 'o', '^'),
             fig: plt.Figure = None, save_to=None, **kwargs):
        nodes = self.nodes.keys()
        arcs = [(i, j) for i in nodes for j in nodes if i != j]

        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(arcs)

        if not fig:
            fig = plt.figure()

        pos = {i: (self.nodes[i].pos_x, self.nodes[i].pos_y) for i in nodes}
        nx.draw(g, pos=pos, nodelist=self.depots, arrows=False, node_color=color[0], node_shape=shape[0], **kwargs)
        nx.draw(g, pos=pos, nodelist=self.customers, arrows=False, node_color=color[1], node_shape=shape[1], **kwargs)
        nx.draw(g, pos=pos, nodelist=self.charging_stations, arrows=False, node_color=color[2], node_shape=shape[2],
                **kwargs)
        if save_to:
            fig.savefig(save_to)
        return fig, g

    def get_networkx(self):
        nodes = self.nodes.keys()
        arcs = [(i, j) for i in nodes for j in nodes if i != j]
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(arcs)
        return g



def from_element_tree(tree):
    _info: ET = tree.find('info')
    _network: ET = tree.find('network')
    _fleet: ET = tree.find('fleet')
    _technologies: ET = _network.find('technologies')

    nodes = {}
    edges = {}
    for _node in _network.find('nodes'):
        node_id = int(_node.get('id'))
        pos_x, pos_y = float(_node.get('pos_x')), float(_node.get('pos_y'))
        typ = int(_node.get('type'))

        if typ == 0:
            node = DepotNode(node_id, pos_x=pos_x, pos_y=pos_y)

        elif typ == 1:
            spent_time = float(_node.get('spent_time'))
            time_window_low = float(_node.get('time_window_low'))
            time_window_upp = float(_node.get('time_window_upp'))
            demand = float(_node.get('demand'))
            node = CustomerNode(node_id, spent_time, demand, pos_x, pos_y, time_window_low, time_window_upp)
        else:
            capacity = int(_node.get('capacity'))
            technology = int(_node.get('technology'))
            _technology = _technologies[technology - 1]
            time_points = tuple([float(bp.get('charging_time')) for bp in _technology])
            soc_points = tuple([float(bp.get('battery_level')) for bp in _technology])
            node = ChargeStationNode(node_id, capacity, pos_x=pos_x, pos_y=pos_y, time_points=time_points,
                                     soc_points=soc_points)
        nodes[node_id] = node

    for _node_from in _network.find('edges'):
        node_from_id = int(_node_from.get('id'))
        d_from = edges[node_from_id] = {}
        for _node_to in _node_from:
            node_to_id = int(_node_to.get('id'))
            if _node_to.get('travel_time') is not None:
                tt = float(_node_to.get('travel_time'))
                ec = float(_node_to.get('energy_consumption'))
                d_from[node_to_id] = Edge(node_from_id, node_to_id, tt, ec)
            else:
                _tt, _ec = _node_to.find('travel_time'), _node_to.find('energy_consumption')
                tt = np.array([float(bp.get('value')) for bp in _tt])
                ec = np.array([float(bp.get('value')) for bp in _ec])
                s = int(_tt[1].get('time_of_day')) - int(_tt[0].get('time_of_day'))
                d_from[node_to_id] = DynamicEdge(node_from_id, node_to_id, s, tt, ec)
    return Network(nodes, edges)


def from_xml(path):
    tree = ET.parse(path)
    return from_element_tree(tree)
