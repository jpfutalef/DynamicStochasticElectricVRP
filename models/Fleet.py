import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

import numpy as np
from numpy import ndarray, zeros

from models.ElectricVehicle import ElectricVehicle
from models.Network import Network
from models.Node import DepotNode, CustomerNode, ChargeStationNode
from models.Edge import Edge, DynamicEdge

RouteVector = Tuple[Tuple[int, ...], Tuple[float, ...]]
RouteDict = Dict[int, Tuple[RouteVector, float, float, float]]


def distance(results, multi: ndarray, b: ndarray):
    return np.sum([dist_fun(multi[i, 0], b[i, 0]) for i, result in enumerate(results) if not result])


def dist_fun(x, y):
    # return np.abs(x - y)
    # return np.sqrt(np.power(x - y, 2))
    # return np.abs(np.power(x - y, 2))
    return (x - y) ** 2


def integer_to_tuple2(k: int):
    val = int(k / 2)
    if k % 2:
        return val, 1
    return val, 0


def theta_vector(init_state, time_vectors, network_size, events_count=None):
    if events_count:
        theta = zeros((network_size, events_count))
    else:
        events_count = sum([len(x[0]) + len(x[1]) for x in time_vectors]) + 1
        theta = zeros((network_size, events_count))

    theta[:, 0] = init_state
    counters = [[TupleIndex([0, 0]), 0] for _ in range(len(time_vectors))]

    for k in range(1, events_count):
        min_t, ind_ev = min([(time_vectors[ind][x[0][1]][x[0][0]], ind) for ind, x in enumerate(counters) if not x[1]])
        if counters[ind_ev][0][1]:
            event = 1
        else:
            event = -1
        node = time_vectors[ind_ev][2][counters[ind_ev][0][0] + counters[ind_ev][0][1]]
        theta[:, k] = theta[:, k - 1]
        theta[node, k] += event
        counters[ind_ev][0] = counters[ind_ev][0] + 1
        if not counters[ind_ev][0][1] and len(time_vectors[ind_ev][2]) - 1 == counters[ind_ev][0][0] + \
                counters[ind_ev][0][1]:
            counters[ind_ev][1] = 1

    return theta


def theta_vector_in_array(init_state, time_vectors, network_size, events_count, op_vector, theta_index) -> None:
    iTheta0 = theta_index
    iTheta1 = theta_index + network_size
    op_vector[iTheta0:iTheta1] = init_state

    counters = [[TupleIndex([0, 0]), 0] for _ in range(len(time_vectors))]

    for k in range(1, events_count):
        iTheta0 += network_size
        iTheta1 += network_size
        min_t, ind_ev = min([(time_vectors[ind][x[0][1]][x[0][0]], ind) for ind, x in enumerate(counters) if not x[1]])

        if counters[ind_ev][0][1]:
            event = 1
        else:
            event = -1

        node = time_vectors[ind_ev][2][counters[ind_ev][0][0] + counters[ind_ev][0][1]]
        op_vector[iTheta0:iTheta1] = op_vector[iTheta0 - network_size:iTheta1 - network_size]
        op_vector[iTheta0 + node] += event
        counters[ind_ev][0] = counters[ind_ev][0] + 1
        if not counters[ind_ev][0][1] and len(time_vectors[ind_ev][2]) - 1 == counters[ind_ev][0][0] + \
                counters[ind_ev][0][1]:
            counters[ind_ev][1] = 1
    return


class TupleIndex(list):
    def __add__(self, other: int):
        i = 2 * self[0] + self[1] + other
        return TupleIndex(integer_to_tuple2(i))


@dataclass
class RealTimeData:
    x1_0: float = 0.0
    x2_0: float = 0.0
    x3_0: float = 0.0


class Fleet:
    vehicles: Dict[int, ElectricVehicle]
    network: Network
    vehicles_to_route: Tuple[int, ...]
    theta_vector: Union[ndarray, None]
    optimization_vector: Union[ndarray, None]
    optimization_vector_indices: Union[Tuple, None]
    realtime_data: Dict[int, RealTimeData]

    def __init__(self, vehicles=None, network=None, vehicles_to_route=None):
        self.set_vehicles(vehicles)
        self.set_network(network)
        self.set_vehicles_to_route(vehicles_to_route)

        self.theta_vector = None
        self.optimization_vector = None
        self.optimization_vector_indices = None

        self.realtime_data = {}

    def set_vehicles(self, vehicles: Dict[int, ElectricVehicle]) -> None:
        self.vehicles = vehicles

    def set_network(self, net: Network) -> None:
        self.network = net

    def set_vehicles_to_route(self, vehicles: List[int]) -> None:
        if vehicles:
            self.vehicles_to_route = tuple(vehicles)

    def set_routes_of_vehicles(self, routes: RouteDict) -> None:
        for id_ev, (route, dep_time, dep_soc, dep_pay) in routes.items():
            self.vehicles[id_ev].set_route(route, dep_time, dep_soc, dep_pay)
            self.vehicles[id_ev].iterate_space(self.network)

    def create_optimization_vector(self) -> ndarray:
        # It is assumed that each EV has a set route by using the ev.set_rout(...) method
        # 0. Preallocate optimization vector
        sum_si = sum([len(self.vehicles[x].route[0]) for x in self.vehicles_to_route])
        length_op_vector = sum_si * (8 + 2 * len(self.network)) - 2 * len(self.vehicles_to_route) * (
                len(self.network) + 1) + len(self.network)
        self.optimization_vector = zeros(length_op_vector)

        # 1. Iterate each ev state to fill their matrices
        iS = 0
        iL = sum_si
        ix1 = iL + sum_si
        ix2 = ix1 + sum_si
        ix3 = ix2 + sum_si
        iD = ix3 + sum_si
        iT = iD + sum_si
        iE = iT + sum_si - len(self.vehicles_to_route)
        iTheta = iE + sum_si - len(self.vehicles_to_route)

        self.optimization_vector_indices = (iS, iL, ix1, ix2, ix3, iD, iT, iE, iTheta)

        t_list = []
        for id_ev in self.vehicles_to_route:
            si = len(self.vehicles[id_ev].route[0])
            self.optimization_vector[iS:iS + si] = self.vehicles[id_ev].route[0]
            self.optimization_vector[iL:iL + si] = self.vehicles[id_ev].route[1]
            self.optimization_vector[ix1:ix1 + si] = self.vehicles[id_ev].state_reaching[0, :]
            self.optimization_vector[ix2:ix2 + si] = self.vehicles[id_ev].state_reaching[1, :]
            self.optimization_vector[ix3:ix3 + si] = self.vehicles[id_ev].state_reaching[2, :]
            self.optimization_vector[iD:iD + si] = self.vehicles[id_ev].charging_times
            self.optimization_vector[iT:iT + si - 1] = self.vehicles[id_ev].travel_times
            self.optimization_vector[iE:iE + si - 1] = self.vehicles[id_ev].energy_consumption

            t_list.append((self.vehicles[id_ev].state_leaving[0, :-1], self.vehicles[id_ev].state_reaching[0, 1:],
                           self.vehicles[id_ev].route[0]))

            iS += si
            iL += si
            ix1 += si
            ix2 += si
            ix3 += si
            iD += si
            iT += si - 1
            iE += si - 1

        # 2. Create theta vector
        init_theta = zeros(len(self.network))
        init_theta[0] = len(self.vehicles)
        theta_vector_in_array(init_theta, t_list, len(self.network), 2 * sum_si - 2 * len(self.vehicles) + 1,
                              self.optimization_vector, iTheta)

        # 3. Create optimization vector
        return self.optimization_vector

    def cost_function(self, w1: float, w2: float, w3: float, w4: float) -> Tuple:
        iS, iL, ix1, ix2, ix3, iD, iT, iE, iTheta = self.optimization_vector_indices
        op_vector = self.optimization_vector
        cost_tt = w1 * np.sum(op_vector[iT:iE])
        cost_ec = w2 * np.sum(op_vector[iE:iTheta])
        cost_chg_op = w3 * np.sum(op_vector[iD:iT])
        cost_chg_cost = w4 * np.sum(op_vector[iL:ix1])
        return cost_tt, cost_ec, cost_chg_op, cost_chg_cost

    def feasible(self) -> (bool, Union[int, float]):
        # Variables to return
        is_feasible = True
        dist = 0

        # Variables from the optimization vector and vehicles
        n_vehicles = len(self.vehicles)
        n_customers = sum([len(vehicle.assigned_customers) for _, vehicle in self.vehicles.items()])
        network = self.network
        sum_si = np.sum(len(vehicle.route[0]) for _, vehicle in self.vehicles.items())
        length_op_vector = len(self.optimization_vector)

        iS, iL, ix1, ix2, ix3, iD, iT, iE, iTheta = self.optimization_vector_indices

        # Amount of rows
        rows = 0

        rows += n_vehicles  # 2.16
        rows += n_customers  # 2.17
        rows += n_customers  # 2.18
        rows += sum_si  # 2.25-1
        rows += sum_si  # 2.25-2
        rows += sum_si  # 2.26-1
        rows += sum_si  # 2.26-2

        # Matrices
        A = np.zeros((rows, length_op_vector))
        b = np.zeros((rows, 1))

        # Start filling
        row = 0

        # 2.16
        si = 0
        for j, vehicle in self.vehicles.items():
            A[row, ix1 + si] = -1.0
            si += len(vehicle.route[0])
            A[row, ix1 + si - 1] = 1.0
            b[row] = vehicle.max_tour_duration
            row += 1

        # 2.17 & 2.18
        si = 0
        for _, vehicle in self.vehicles.items():
            for k, (Sk, Lk) in enumerate(zip(vehicle.route[0], vehicle.route[1])):
                if network.isCustomer(Sk):
                    A[row, ix1 + si + k] = -1.0
                    b[row] = -network.nodes[Sk].timeWindowDown
                    row += 1

                    A[row, ix1 + si + k] = 1.0
                    b[row] = network.nodes[Sk].timeWindowUp - network.spent_time(Sk, None, None)
                    row += 1
            si += len(vehicle.route[0])

        # 2.25-1 & 2.25-2
        si = 0
        for _, vehicle in self.vehicles.items():
            for k, (Sk, Lk) in enumerate(zip(vehicle.route[0], vehicle.route[1])):
                A[row, ix2 + si + k] = -1.0
                b[row] = -vehicle.alpha_down
                row += 1

                A[row, ix2 + si + k] = 1.0
                b[row] = vehicle.alpha_up
                row += 1
            si += len(vehicle.route[0])

        # 2.26-1 & 2.26-2
        si = 0
        for _, vehicle in self.vehicles.items():
            for k, (Sk, Lk) in enumerate(zip(vehicle.route[0], vehicle.route[1])):
                A[row, ix2 + si + k] = -1.0
                b[row] = -vehicle.alpha_down + Lk
                row += 1

                A[row, ix2 + si + k] = 1.0
                b[row] = vehicle.alpha_up - Lk
                row += 1
            si += len(vehicle.route[0])

        # Check
        multi = np.matmul(A, np.vstack(self.optimization_vector.T))
        boolList = multi <= b
        for result in boolList:
            if not result:
                dist = distance(boolList, multi, b)
                is_feasible = False
                break

        return is_feasible, dist

    # From XML
    def from_xml(self, path):
        # Open XML file
        tree = ET.parse(path)
        _info: ET = tree.find('info')
        _network: ET = tree.find('network')
        _fleet: ET = tree.find('fleet')
        _technologies: ET = _network.find('technologies')

        # Network data
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
                tt = _node_to.get('travel_time')
                ec = _node_to.get('energy_consumption')
                d_from[node_to_id] = Edge(node_from_id, node_to_id, tt, ec)

        self.set_network(Network(nodes, edges))

        # Fleet data
        vehicles = {}
        for _vehicle in _fleet:
            ev_id = int(_vehicle.get('id'))
            max_tour_duration = float(_vehicle.get('max_tour_duration'))
            alpha_up = float(_vehicle.get('alpha_up'))
            alpha_down = float(_vehicle.get('alpha_down'))
            battery_capacity = float(_vehicle.get('battery_capacity'))
            max_payload = float(_vehicle.get('max_payload'))
            vehicles[ev_id] = ElectricVehicle(ev_id, alpha_up, alpha_down, max_tour_duration, battery_capacity,
                                              max_payload)

        self.set_vehicles(vehicles)

    # Realtime tools
    def update_from_xml(self, path, do_network=False, set_customers=False, realtime=False):
        # Open XML file
        tree = ET.parse(path)
        _info: ET = tree.find('info')
        _network: ET = tree.find('network')
        _fleet: ET = tree.find('fleet')
        _technologies: ET = _network.find('technologies')

        # Network data
        if do_network:
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
                    tt = _node_to.get('travel_time')
                    ec = _node_to.get('energy_consumption')
                    d_from[node_to_id] = Edge(node_from_id, node_to_id, tt, ec)

            self.set_network(Network(nodes, edges))

        # Fleet data
        for _vehicle in _fleet:
            ev_id = int(_vehicle.get('id'))
            if set_customers:
                _customers_to_visit = _vehicle.find('assigned_customers')
                customers_to_visit = tuple(int(x.get('id')) for x in _customers_to_visit)
                self.vehicles[ev_id].set_customers_to_visit(customers_to_visit)

            if realtime:
                _previous_sequence = _vehicle.find('previous_sequence')
                if _previous_sequence:  # TODO check if empty equals False
                    _critical_point = _vehicle.find('critical_point')
                    k = int(_critical_point.get('k'))
                    previous_sequence = [tuple(int(x.get('sk')) for x in _previous_sequence),
                                         tuple(float(x.get('lk')) for x in _previous_sequence)]
                    new_customers = tuple(x for x in previous_sequence[0][k+1:] if self.network.isCustomer(x))
                    self.vehicles[ev_id].set_customers_to_visit(new_customers)

                    x1_0 = float(_critical_point.get('x1'))
                    x2_0 = float(_critical_point.get('x2'))
                    x3_0 = sum([self.network.demand(x) for x in self.vehicles[ev_id].assigned_customers])
                    self.realtime_data[ev_id] = RealTimeData(x1_0, x2_0, x3_0)
