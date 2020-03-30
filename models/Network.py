from typing import Tuple, Dict, Union

from models.Edge import Edge, DynamicEdge
from models.Node import CustomerNode, ChargeStationNode, DepotNode

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
