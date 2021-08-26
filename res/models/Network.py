import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Tuple, Union, List

from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import res.models.Edge as Edge
import res.models.Node as Node
import res.tools.IOTools as IOTools

NodeDict = Dict[int, Union[Node.BaseNode, Node.DepotNode, Node.CustomerNode, Node.ChargingStationNode]]
EdgeDict = Dict[int, Dict[int, Union[Edge.DynamicEdge, Edge.GaussianEdge]]]

'''
UTILITIES
'''


class TupleIndex(list):
    def __add__(self, other: int):
        i = 2 * self[0] + self[1] + other
        return TupleIndex(integer_to_tuple2(i))


def integer_to_tuple2(k: int):
    val = int(k / 2)
    if k % 2:
        return val, 1
    return val, 0


def theta_matrix(matrix, time_vectors, events_count):
    counters = [[TupleIndex([0, 0]), 0] for _ in range(len(time_vectors))]
    for k in range(1, events_count):
        min_t, ind_ev = min([(time_vectors[ind][x[0][1]][x[0][0]], ind) for ind, x in enumerate(counters) if not x[1]])
        event = 1 if counters[ind_ev][0][1] else -1
        node = time_vectors[ind_ev][2][counters[ind_ev][0][0] + counters[ind_ev][0][1]]
        matrix[:, k] = matrix[:, k - 1]
        matrix[node, k] += event
        counters[ind_ev][0] = counters[ind_ev][0] + 1
        if not counters[ind_ev][0][1] and len(time_vectors[ind_ev][2]) - 1 == counters[ind_ev][0][0] + \
                counters[ind_ev][0][1]:
            counters[ind_ev][1] = 1


"""
CLASS DEFINITIONS
"""


@dataclass
class Network:
    nodes: NodeDict
    edges: EdgeDict
    depots: Tuple[int, ...] = tuple()
    customers: Tuple[int, ...] = tuple()
    charging_stations: Tuple[int, ...] = tuple()
    type: str = None

    def __post_init__(self):
        [self.add_to_ids(node) for node in self.nodes.values()]
        self.type = self.__class__.__name__

    def __len__(self):
        return len(self.nodes)

    def set_nodes(self, node_collection: NodeDict):
        self.nodes = node_collection
        self.depots = ()
        self.customers = ()
        self.charging_stations = ()

        for node in node_collection.values():
            self.add_to_ids(node)

    def set_edges(self, edge_collection: EdgeDict):
        del self.edges
        self.edges = edge_collection

    def add_to_ids(self, node: Node.BaseNode):
        if node.is_depot():
            self.depots += (node.id,)
        elif node.is_customer():
            self.customers += (node.id,)
        else:
            self.charging_stations += (node.id,)

    def add_node(self, node: Node.BaseNode):
        self.nodes[node.id] = node
        self.add_to_ids(node)

    def add_nodes(self, node_collection: NodeDict):
        for node in node_collection.values():
            self.add_node(node)

    def v(self, node_from: int, node_to: int, time_of_day: float) -> Union[float, Tuple[float, float]]:
        return self.edges[node_from][node_to].get_velocity(time_of_day)

    def spent_time(self, node: int, init_soc: float, soc_increment, eta=None):
        return self.nodes[node].service_time(init_soc, soc_increment, eta)

    def demand(self, node: int):
        return self.nodes[node].demand

    def arc_length(self, node_from: int, node_to: int) -> float:
        return self.edges[node_from][node_to].length

    def time_window_low(self, node: int):
        return self.nodes[node].time_window_low

    def time_window_upp(self, node: int):
        return self.nodes[node].time_window_upp

    def is_depot(self, node: int):
        return self.nodes[node].is_depot()

    def is_customer(self, node: int):
        return self.nodes[node].is_customer()

    def is_charging_station(self, node: int):
        return self.nodes[node].is_charge_station()

    def drop_time_windows(self):
        for node in self.nodes.values():
            node.time_window_upp = np.infty
            node.time_window_low = -np.infty

    def draw(self, color=('lightskyblue', 'limegreen', 'goldenrod'), shape=('s', 'o', '^'),
             fig: plt.Figure = None, save_to=None, **kwargs):
        nodes = self.nodes.keys()
        nodes_id = {i: i for i, node in self.nodes.items()}
        arcs = [(i, j) for i in nodes for j in nodes if i != j]

        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(arcs)

        if not fig:
            fig = plt.figure()

        pos = {i: (self.nodes[i].pos_x, self.nodes[i].pos_y) for i in nodes}
        nx.draw(g, pos=pos, nodelist=self.depots, arrows=False, node_color=color[0], node_shape=shape[0],
                labels=nodes_id, **kwargs)
        nx.draw(g, pos=pos, nodelist=self.customers, arrows=False, node_color=color[1], node_shape=shape[1],
                labels=nodes_id, **kwargs)
        nx.draw(g, pos=pos, nodelist=self.charging_stations, arrows=False, node_color=color[2], node_shape=shape[2],
                labels=nodes_id, **kwargs)
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

    def xml_tree(self):
        _network = ET.Element('network')
        _nodes = ET.SubElement(_network, 'nodes')
        _edges = ET.SubElement(_network, 'edges')

        _network.set('type', self.type)

        for node in self.nodes.values():
            _nodes.append(node.xml_element())
            _node_from = ET.SubElement(_edges, 'node_from', attrib={'id': str(node.id)})
            for node_to in self.nodes.values():
                _node_to = self.edges[node.id][node_to.id].xml_element()
                _node_from.append(_node_to)
        return _network

    def write_xml(self, filepath: str, print_pretty=False):
        tree = self.xml_tree()
        if print_pretty:
            IOTools.write_pretty_xml(filepath, tree)
        else:
            ET.ElementTree(tree).write(filepath)

    @classmethod
    def from_xml_element(cls, element: ET.Element):
        nodes = {}
        for _node in element.find('nodes'):
            node = Node.from_xml_element(_node)
            nodes[node.id] = node

        edges = {}
        for _node_from in element.find('edges'):
            node_from = int(_node_from.get('id'))
            dict_from = edges[node_from] = {}
            for _node_to in _node_from:
                edge = Edge.from_xml_element(_node_to, node_from)
                dict_from[edge.node_to] = edge

        return cls(nodes, edges)

    @classmethod
    def from_xml(cls, xml_file: Union[str, ET.Element], return_etree=False):
        if type(xml_file) == str:
            tree = ET.parse(xml_file).getroot()
        else:
            tree = xml_file

        is_instance = True if tree.tag == 'instance' else False
        _network = tree.find('network') if is_instance else tree

        if return_etree:
            return cls.from_xml_element(_network), tree
        else:
            return cls.from_xml_element(_network)


@dataclass
class DeterministicCapacitatedNetwork(Network):
    theta_matrix: Union[None, np.ndarray] = None
    theta_matrix_container: Union[None, np.ndarray] = None
    penalization: float = 0.0

    def __post_init__(self):
        super(DeterministicCapacitatedNetwork, self).__post_init__()
        num_events = 2 * len(self)
        self.theta_matrix_container = np.zeros((len(self), num_events))

    def iterate_cs_capacities(self, init_theta: np.ndarray, time_vectors: List, fleet_size: int):
        """
        Calculates occupation matrix.
        @param init_theta: Vector containing the number of vehicles at each node, at the moment of calculation.
        @param time_vectors: List containing information about routes, and arrival and departure times. Each element in
         this list is a tuple (Time arrival_i, Time_departure_i, Route_i), for EV i.
        @param fleet_size: the number of EVs in the fleets
        @return: None. The matrix is then accessible via NETWORK_INSTANCE.theta_matrix
        """
        sum_si = sum([len(t[2]) for t in time_vectors])
        num_events = 2 * sum_si - 2 * fleet_size + 1
        if num_events > self.theta_matrix_container.shape[1]:
            diff = num_events - self.theta_matrix_container.shape[1]
            self.theta_matrix_container = np.append(self.theta_matrix_container, np.zeros((len(self), diff)), axis=1)
        # self.theta_matrix_container.fill(0)
        self.theta_matrix_container[:, 0] = init_theta
        theta_matrix(self.theta_matrix_container, time_vectors, num_events)
        self.theta_matrix = self.theta_matrix_container[:, :num_events]

    def noise_gain(self, factor: float):
        # TODO test it
        for i in self.edges.values():
            for j in i.values():
                j.velocity_deviation *= factor

    def check_feasibility(self):
        self.penalization = 0.0
        self.constraint_cs_capacities()
        return self.penalization

    def constraint_cs_capacities(self):
        p = 0.0
        for cs_id in self.charging_stations:
            row = self.theta_matrix[cs_id, :]
            capacity = self.nodes[cs_id].capacity
            p += sum([val - capacity if val - capacity > 0 else 0 for val in row])
        self.penalization += p


"""
CAPACITATED GAUSSIAN NETWORK - DEFINITION
"""


def evaluate_saturation(cs_id: int, num_customers: int, constraints: List[List],
                        probability_matrices: Dict[int, np.ndarray],
                        result_container: np.ndarray, num_depots: int = 1):
    """
    Evaluates the saturation probability of a CS, given an operational behavior of EVs
    @param cs_id: ID of the CS to evaluate
    @param num_customers: number of customers in the network
    @param constraints:
    @param probability_matrices: matrix
    @param result_container:
    @param num_depots:
    @return:
    """
    row = cs_id - num_depots - num_customers
    for k in range(len(result_container)):
        probability = sum([m[row, k] if constraints[k] else 1 - m[row, k] for m in probability_matrices.values()])
        result_container[k] = probability


def probability_saturation(description: List[Tuple], probabilities_in_nodes: List[np.ndarray]):
    return sum([single_saturation_probability(d, probabilities_in_nodes) for d in description])


def single_saturation_probability(description: Tuple, probabilities_in_nodes: List[np.ndarray]):
    if len(description) != len(probabilities_in_nodes):
        raise ValueError("Length of probability descriptions does not match number of calculated probabilities")
    p = 1
    for i, prob in zip(description, probabilities_in_nodes):
        factor = prob if i else 1 - prob
        p *= factor
    return p


def heuristic_1(self):
    num_of_evs_that_recharge = sum([1 if ev.visits_a_cs else 0 for ev in self.vehicles.values()])


def heuristic_2(self):
    return


@dataclass
class CapacitatedGaussianNetwork(Network):
    sample_time_occupation_probability: float = 5.
    cdf_container: Union[np.ndarray, None] = None
    probability_of_occupation: Union[np.ndarray, None] = None

    def __post_init__(self):
        self.setup_saturation_container(self.sample_time_occupation_probability)

    def setup_saturation_container(self, sample_time: float = None):
        if sample_time:
            self.sample_time_occupation_probability = sample_time
        self.probability_of_occupation = np.zeros(
            (len(self.charging_stations), 24 * 60 / self.sample_time_occupation_probability))

    def constraint_cs_capacities(self):
        p = 0.0
        self.penalization += p

    def check_feasibility(self):
        self.penalization = 0.0
        self.constraint_cs_capacities()
        return self.penalization


def from_xml_element(element: ET.Element):
    t = element.get('type')
    cls = globals()[t]
    return cls.from_xml_element(element)


def from_xml(filepath: Union[str, Path]):
    element = ET.parse(filepath).getroot()
    if element.tag == 'network':
        _network = element
    else:
        _network = element.find('network')
    t = _network.get('type')
    cls = globals()[t]
    return cls.from_xml(element)
