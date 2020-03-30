from dataclasses import dataclass
from typing import Union, Tuple, Dict

import numpy as np
from numpy import ndarray, array

from models.Edge import Edge
from models.Network import Network, DynamicNetwork
from models.Node import DepotNode, CustomerNode, ChargeStationNode

RouteVector = Tuple[Tuple[int, ...], Tuple[float, ...]]
RouteDict = Dict[int, Tuple[RouteVector, float, float, float]]


def F_recursive(k: int, route: RouteVector, state_reaching_matrix: ndarray, state_leaving_matrix: ndarray,
                tt_array: ndarray, ec_array: ndarray, c_op_array: ndarray, network: Network):
    if k == 0:
        return state_reaching_matrix[:, 0]
    else:
        tt = network.t(route[0][k - 1], route[0][k], time_of_day=state_leaving_matrix[0, k - 1])
        ec = network.e(route[0][k - 1], route[0][k], payload=state_leaving_matrix[2, k - 1],
                       time_of_day=state_leaving_matrix[0, k - 1])
        d = network.demand(route[0][k])
        u1 = array((tt, -ec, -d))
        state_reaching_matrix[:, k] = F_recursive(k - 1, route, state_reaching_matrix, state_leaving_matrix,
                                                  tt_array, ec_array, c_op_array, network) + u1

        spent_time = network.spent_time(route[0][k], state_reaching_matrix[1, k], route[1][k])
        u2 = array((spent_time, route[1][k], 0))
        state_leaving_matrix[:, k] = state_reaching_matrix[:, k] + u2

        tt_array[0, k - 1] = tt
        ec_array[0, k - 1] = ec

        if network.isChargingStation(route[0][k]):  # TODO si el primero es cs, incluirlo?
            c_op_array[0, k] = spent_time

        return state_leaving_matrix[:, k]


def cost_arc(item):
    cost = 0
    for node_from, d in item.items():
        for node_to, c in d.items():
            cost += c
    return cost


def cost_node(item):
    cost = 0
    for node, c in item:
        cost += c
    return cost


def feasible(x: np.ndarray, vehicles: dict):
    """
    Checks feasibility of the optimization vector x. It's been assumed all EVs have already ran the
    ev.iterate() method.
    :param x: optimization vector
    :param vehicles: dict with vehicles info by id
    :return: Tuple (feasibility, distance) where feasibility=True and distance=0 if x is feasible; otherwise,
    feasibility=False and distance>0 if x isn't feasible. Distance is the squared accumulated distance.
    """
    vehicle: ElectricVehicle

    # Variables to return
    is_feasible = True
    dist = 0

    # Variables from the optimization vector and vehicles
    n_vehicles = len(vehicles)
    n_customers = np.sum([vehicle.ni for _, vehicle in vehicles.items()])
    network = vehicles[0].network
    sum_si = np.sum(len(vehicle.node_sequence) for _, vehicle in vehicles.items())
    lenght_op_vector = len(x)

    i_S, i_L, i_x1, i_x2, i_x3, i_theta = 0, sum_si, 2 * sum_si, 3 * sum_si, 4 * sum_si, 5 * sum_si

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
    A = np.zeros((rows, lenght_op_vector))
    b = np.zeros((rows, 1))

    # Start filling
    row = 0

    # 2.16
    si = 0
    for j, vehicle in vehicles.items():
        A[row, i_x1 + si] = -1.0
        si += len(vehicle.node_sequence)
        A[row, i_x1 + si - 1] = 1.0
        b[row] = vehicle.max_tour_duration
        row += 1

    # 2.17 & 2.18
    si = 0
    for _, vehicle in vehicles.items():
        for k, Sk in enumerate(vehicle.node_sequence):
            if network.isCustomer(Sk):
                A[row, i_x1 + si + k] = -1.0
                b[row] = -network.nodes[Sk]['attr'].timeWindowDown
                row += 1

                A[row, i_x1 + si + k] = 1.0
                b[row] = network.nodes[Sk]['attr'].timeWindowUp - network.spent_time(Sk)
                row += 1
        si += len(vehicle.node_sequence)

    # 2.25-1 & 2.25-2
    si = 0
    for _, vehicle in vehicles.items():
        for k, Sk in enumerate(vehicle.node_sequence):
            A[row, i_x2 + si + k] = -1.0
            b[row] = -vehicle.alpha_down
            row += 1

            A[row, i_x2 + si + k] = 1.0
            b[row] = vehicle.alpha_up
            row += 1
        si += len(vehicle.node_sequence)

    # 2.26-1 & 2.26-2
    si = 0
    for _, vehicle in vehicles.items():
        for k, Sk in enumerate(vehicle.node_sequence):
            A[row, i_x2 + si + k] = -1.0
            b[row] = -vehicle.alpha_down + vehicle.charging_sequence[k]
            row += 1

            A[row, i_x2 + si + k] = 1.0
            b[row] = vehicle.alpha_up - vehicle.charging_sequence[k]
            row += 1
        si += len(vehicle.node_sequence)

    # Check
    mult = np.matmul(A, x)
    boolList = mult <= b
    for result in boolList:
        if not result:
            dist = distance(boolList, mult, b, vehicles)
            is_feasible = False
            break

    return is_feasible, dist


def distance(results, mult, b, vehicles):
    return np.sum([dist_fun(mult[i, 0], b[i, 0]) for i, result in enumerate(results) if result])


def dist_fun(x, y):
    # return np.abs(x - y)
    # return np.sqrt(np.power(x - y, 2))
    # return np.abs(np.power(x - y, 2))
    return (x - y) ** 2


@dataclass
class ElectricVehicle:
    vid: int
    # Attribs
    alpha_up: Union[int, float] = 80.
    alpha_down: Union[int, float] = 40.
    max_tour_duration: Union[int, float] = 300.
    battery_capacity: Union[int, float] = 200.
    max_payload: Union[int, float] = 2.

    # Required by model
    assigned_customers: Tuple[int, ...] = None
    x1_0: float = 0.0
    x2_0: float = 100.0
    x3_0: float = 0.0
    route: RouteVector = None
    travel_times: ndarray = None
    energy_consumption: ndarray = None
    charging_times: ndarray = None
    state_reaching: ndarray = None
    state_leaving: ndarray = None

    def set_customers_to_visit(self, new_customers: Tuple[int, ...]):
        self.assigned_customers = new_customers

    def set_route(self, new_route: RouteVector, depart_time: float, depart_soc: float, depart_payload: float,
                  stochastic=False):
        self.route = new_route
        self.x1_0 = depart_time
        self.x2_0 = depart_soc
        self.x3_0 = depart_payload

        if stochastic:
            # TODO define sizes properly for the stochastic case
            len_route = len(new_route[0])
            size_state_matrix = (3, len_route)

            size_tt = (1, len_route - 1)
            size_ec = (1, len_route - 1)
            size_c_op = (1, len_route)

            init_state = (self.x1_0, self.x2_0, self.x3_0)
        else:
            len_route = len(new_route[0])
            size_state_matrix = (3, len_route)
            size_tt = (1, len_route - 1)
            size_ec = (1, len_route - 1)
            size_c_op = (1, len_route)

            init_state = (self.x1_0, self.x2_0, self.x3_0)

        self.state_reaching = np.zeros(size_state_matrix)
        self.state_leaving = np.zeros(size_state_matrix)
        self.state_reaching[:, 0] = np.array(init_state)
        self.state_leaving[:, 0] = np.array(init_state)

        self.travel_times = np.zeros(size_tt)
        self.energy_consumption = np.zeros(size_ec)
        self.charging_times = np.zeros(size_c_op)

    def iterate_space(self, network: Union[Network, DynamicNetwork]):
        k = len(self.route[0]) - 1
        F_recursive(k, self.route, self.state_reaching, self.state_leaving, self.travel_times, self.energy_consumption,
                    self.charging_times, network)

