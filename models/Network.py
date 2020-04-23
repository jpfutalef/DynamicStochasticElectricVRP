from typing import Tuple, Dict, Union

from models.Edge import Edge, DynamicEdge
from models.Node import CustomerNode, ChargeStationNode, DepotNode
import xml.etree.ElementTree as ET

from numpy import array

NodeDict = Dict[int, Union[CustomerNode, ChargeStationNode, DepotNode]]
EdgeDict = Dict[int, Dict[int, Edge]]
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

    def add_to_ids(self, node: NodeType):
        if node.isDepot():
            self.depots = tuple(list(self.depots) + [node.id])

        elif node.isCustomer():
            self.customers = tuple(list(self.customers) + [node.id])

        else:
            self.charging_stations = tuple(list(self.charging_stations) + [node.id])

    def add_node(self, node: NodeType):
        self.nodes[node.id] = node
        self.add_to_ids(node)

    def add_nodes(self, node_collection: NodeDict):
        for node in node_collection.values():
            self.add_node(node)

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

    def t(self, node_from: int, node_to: int, time_of_day=0.0) -> Union[float, int]:
        return self.edges[node_from][node_to].get_travel_time()

    def e(self, node_from: int, node_to: int, payload=0.0, time_of_day=0.0) -> Union[float, int]:
        return self.edges[node_from][node_to].get_energy_consumption(payload)

    def spent_time(self, node: int, p, q):
        return self.nodes[node].spentTime(p, q)

    def demand(self, node: int):
        return self.nodes[node].requiredDemand()

    def isDepot(self, node: int):
        return self.nodes[node].isDepot()

    def isCustomer(self, node: int):
        return self.nodes[node].isCustomer()

    def isChargingStation(self, node: int):
        return self.nodes[node].isChargeStation()

    def draw(self):
        pass


class DynamicNetwork(Network):
    edges: DynamicEdgeDict
    sample_time: Union[int, float]

    def __init__(self, network_nodes: NodeDict = None, network_edges: DynamicEdgeDict = None,
                 samp_time: Union[int, float] = 10):
        super().__init__(network_nodes, network_edges)
        self.sample_time = samp_time

    def t(self, node_from: int, node_to: int, time_of_day=0.0) -> Union[float, int]:
        return self.edges[node_from][node_to].get_travel_time(time_of_day)

    def e(self, node_from: int, node_to: int, payload=0.0, time_of_day=0.0) -> Union[float, int]:
        return self.edges[node_from][node_to].get_energy_consumption(payload, time_of_day)


def from_element_tree(tree):
    _info: ET = tree.find('info')
    _network: ET = tree.find('network')
    _fleet: ET = tree.find('fleet')
    _technologies: ET = _network.find('technologies')

    nodes = {}
    edges = {}
    for _node in _network.find('nodes'):
        node_id = int(_node.get('id'))
        pos = (float(_node.get('cx')), float(_node.get('cy')))
        typ = int(_node.get('type'))

        if typ == 0:
            node = DepotNode(node_id, pos=pos)

        elif typ == 1:
            spent_time = float(_node.get('spent_time'))
            time_window_low = float(_node.get('time_window_low'))
            time_window_upp = float(_node.get('time_window_upp'))
            demand = float(_node.get('demand'))
            node = CustomerNode(node_id, spent_time, demand, time_window_upp, time_window_low, pos=pos)
        else:
            capacity = int(_node.get('capacity'))
            technology = int(_node.get('technology'))
            _technology = _technologies[technology - 1]
            time_points = tuple([float(bp.get('charging_time')) for bp in _technology])
            soc_points = tuple([float(bp.get('battery_level')) for bp in _technology])
            node = ChargeStationNode(node_id, capacity, time_points, soc_points, pos=pos)
        nodes[node_id] = node

    for _node_from in _network.find('edges'):
        node_from_id = int(_node_from.get('id'))
        d_from = edges[node_from_id] = {}
        for _node_to in _node_from:
            node_to_id = int(_node_to.get('id'))
            tt = float(_node_to.get('travel_time'))
            ec = float(_node_to.get('energy_consumption'))
            d_from[node_to_id] = Edge(node_from_id, node_to_id, tt, ec)

    return Network(nodes, edges)


def from_xml(path):
    tree = ET.parse(path)
    return from_element_tree(tree)
