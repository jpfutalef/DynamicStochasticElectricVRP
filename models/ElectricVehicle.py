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
        u1 = array((tt, -ec, 0))
        state_reaching_matrix[:, k] = F_recursive(k - 1, route, state_reaching_matrix, state_leaving_matrix,
                                                  tt_array, ec_array, c_op_array, network) + u1

        spent_time = network.spent_time(route[0][k], state_reaching_matrix[1, k], route[1][k])
        u2 = array((spent_time, route[1][k], -d))
        state_leaving_matrix[:, k] = state_reaching_matrix[:, k] + u2

        tt_array[0, k - 1] = tt
        ec_array[0, k - 1] = ec

        if network.isChargingStation(route[0][k]):  # TODO si el primero es cs, incluirlo?
            c_op_array[0, k] = spent_time

        return state_leaving_matrix[:, k]


@dataclass
class ElectricVehicle:
    # Attribs that must be given
    vid: int
    alpha_up: Union[int, float]
    alpha_down: Union[int, float]
    max_tour_duration: Union[int, float]
    battery_capacity: Union[int, float]
    max_payload: Union[int, float]

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
