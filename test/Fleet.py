from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

from models.ElectricVehicle import ElectricVehicle
from models.Network import Network
from models.Node import DepotNode, CustomerNode, ChargeStationNode
from models.Edge import Edge

import numpy as np
from numpy import ndarray, zeros, array
import xml.etree.ElementTree as ET

RouteVector = Tuple[Tuple[int, ...], Tuple[float, ...]]
RouteDict = Dict[int, Tuple[RouteVector, float, float, float]]


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
    iTheta1 = theta_index+network_size
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
        op_vector[iTheta0:iTheta1] = op_vector[iTheta0-network_size:iTheta1-network_size]
        op_vector[iTheta0+node] += event
        counters[ind_ev][0] = counters[ind_ev][0] + 1
        if not counters[ind_ev][0][1] and len(time_vectors[ind_ev][2]) - 1 == counters[ind_ev][0][0] + counters[ind_ev][0][1]:
            counters[ind_ev][1] = 1
    return


class TupleIndex(list):
    def __add__(self, other: int):
        i = 2 * self[0] + self[1] + other
        return TupleIndex(integer_to_tuple2(i))


class Fleet:
    vehicles: Dict[int, ElectricVehicle]
    network: Network
    vehicles_to_route: Tuple[int, ...]
    theta_vector: Union[ndarray, None]
    optimization_vector: Union[ndarray, None]
    optimization_vector_indices: Union[Tuple, None]

    def __init__(self, vehicles, network=None, vehicles_to_route=None):
        self.vehicles = vehicles
        self.set_network(network)
        self.set_vehicles_to_route(vehicles_to_route)

        self.theta_vector = None
        self.optimization_vector = None
        self.optimization_vector_indices = None

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
        theta_vector_in_array(init_theta, t_list, len(self.network), 2*sum_si-2*len(self.vehicles)+1,
                              self.optimization_vector, iTheta)

        # 3. Create optimization vector
        return self.optimization_vector

    def cost_function(self, w1: float, w2: float, w3: float, w4: float) -> Tuple:
        iS, iL, ix1, ix2, ix3, iD, iT, iE, iTheta = self.optimization_vector_indices
        op_vector = self.optimization_vector
        cost_tt = w1*np.sum(op_vector[iT:iE])
        cost_ec = w2*np.sum(op_vector[iE:iTheta])
        cost_chg_op = w3*np.sum(op_vector[iD:iT])
        cost_chg_cost = w4*np.sum(op_vector[iL:ix1])
        return cost_tt, cost_ec, cost_chg_op, cost_chg_cost

    # From XML
    def from_xml(self, path, net):
        # Open XML file
        tree = ET.parse(path)
        _info = tree.find('info')
        _network = tree.find('network')
        _fleet = tree.find('fleet')

        attrib = {}
        for _attrib in _fleet.find('vehicle_attributes'):
            attrib[_attrib.tag] = float(_attrib.text)

        print('EV attributes:', attrib, '\n')

        # instantiate
        numVehicles = int(_fleet.find('fleet_size').text)
        vehicles = {}
        for id_car in range(numVehicles):
            ev = vehicles[id_car] = ElectricVehicle(id_car, net, **attrib)

        return vehicles

    # Realtime tools


if __name__ == '__main__':
    # Nodes
    depot = DepotNode(0)
    customer1 = CustomerNode(1, 15, .4)
    customer2 = CustomerNode(2, 15, .4)
    customer3 = CustomerNode(3, 15, .4)
    customer4 = CustomerNode(4, 15, .4)
    charge_station1 = ChargeStationNode(5)

    nodes = {0: depot, 1: customer1, 2: customer2, 3: customer3, 4: customer4, 5: charge_station1}

    # Edges
    edges = {}
    for node_from in nodes.keys():
        n_from = edges[node_from] = {}
        for node_to in nodes.keys():
            if node_from == node_to:
                tt = 0
                ec = 0
            else:
                tt = 15.0
                ec = 18.0
            n_from[node_to] = Edge(node_from, node_to, tt, ec)

    # Create a static network
    print('***** STATIC NETWORK *****')
    network = Network(nodes, edges)

    # Now the EVs
    ev1 = ElectricVehicle(1)
    ev2 = ElectricVehicle(2)

    ev1.set_customers_to_visit((1, 2))
    ev2.set_customers_to_visit((3, 4))

    # The fleet
    fleet = Fleet(network, {1: ev1, 2: ev2}, (1, 2))

    # Routes and initial conditions
    x1_0_1 = 10 * 60
    x2_0_1 = 80
    x3_0_1 = sum([network.nodes[i].requiredDemand() for i in ev1.assigned_customers])

    x1_0_2 = 8 * 60
    x2_0_2 = 80
    x3_0_2 = sum([network.nodes[i].requiredDemand() for i in ev2.assigned_customers])

    route1 = ((0, 1, 2, 0), (0, 0, 0, 0))
    route2 = ((0, 3, 5, 4, 0), (0, 0, 10.5, 0, 0))

    routes = {1: (route1, x1_0_1, x2_0_1, x3_0_1), 2: (route2, x1_0_2, x2_0_2, x3_0_2)}

    # Set routes and iterate
    fleet.set_routes_of_vehicles(routes)

    # Get optimization vector
    op_vector = fleet.create_optimization_vector()

    a = 1
