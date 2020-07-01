from dataclasses import dataclass
from typing import Union, Tuple, Dict, NamedTuple, List

import numpy as np
from numpy import ndarray, array
import xml.etree.ElementTree as ET

from models.Network import Network
from models.BatteryDegradation import eta as eta_fun

RouteVector = Tuple[Tuple[int, ...], Tuple[float, ...]]
RouteDict = Dict[int, Tuple[RouteVector, float, float, float]]


class InitialCondition(NamedTuple):
    S0: int
    L0: float
    x1_0: float
    x2_0: float
    x3_0: float


def F_recursive(k: int, route: RouteVector, state_reaching_matrix: ndarray, state_leaving_matrix: ndarray,
                tt_array: ndarray, ec_array: ndarray, c_op_array: ndarray, ev_weight: float, network: Network):
    if k == 0:
        return state_reaching_matrix[:, 0]
    else:
        D = network.demand(route[0][k])
        state_leaving_matrix[2, k - 1] = state_leaving_matrix[2, k] + D

        time_of_day = state_leaving_matrix[2, k - 1]
        payload_weight = state_leaving_matrix[2, k - 1]
        t_ij = network.t(route[0][k - 1], route[0][k], time_of_day)
        e_ij = network.e(route[0][k - 1], route[0][k], payload_weight, ev_weight, time_of_day)
        u1 = array((t_ij, -e_ij, 0))

        state_reaching_matrix[:, k] = F_recursive(k - 1, route, state_reaching_matrix, state_leaving_matrix,
                                                  tt_array, ec_array, c_op_array, ev_weight, network) + u1

        time_in_node = network.spent_time(route[0][k], state_reaching_matrix[1, k], route[1][k])
        Lk = route[1][k]
        u2 = array((time_in_node, Lk))
        state_leaving_matrix[0:2, k] = state_reaching_matrix[0:2, k] + u2

        tt_array[0, k - 1] = t_ij
        ec_array[0, k - 1] = e_ij

        if network.isChargingStation(route[0][k]):
            c_op_array[0, k] = time_in_node

        return state_leaving_matrix[:, k]


def F_step(route: RouteVector, state_reaching_matrix: ndarray, state_leaving_matrix: ndarray,
           tt_array: ndarray, ec_array: ndarray, c_op_array: ndarray, ev_weight: float, eta: list, network: Network):
    Sk, Lk = route[0], route[1]
    eta_init_index = 0
    for k, (Sk0, Lk0, Sk1, Lk1) in enumerate(zip(Sk[:-1], Lk[:-1], Sk[1:], Lk[1:]), 1):
        # TODO add waiting time
        departure_time = state_leaving_matrix[0, k - 1]
        payload = state_leaving_matrix[2, k - 1]

        coef = np.prod(eta)
        coef = 1. if -.01 < coef <.01 else coef
        tij = network.t(Sk0, Sk1, departure_time)
        eij = network.e(Sk0, Sk1, payload, ev_weight, departure_time)/coef
        tt_array[0, k - 1] = tij
        ec_array[0, k - 1] = eij

        state_reaching_matrix[:, k] = state_leaving_matrix[:, k - 1] + np.array([tij, -eij, 0])

        tj = network.spent_time(Sk1, state_reaching_matrix[1, k], Lk1)
        dj = network.demand(Sk1)

        state_leaving_matrix[:, k] = state_reaching_matrix[:, k] + np.array([tj, Lk1, -dj])

        if network.isChargingStation(Sk1):
            c_op_array[0, k] = tj
            soch, socl = state_leaving_matrix[1, eta_init_index], state_reaching_matrix[1, k]
            eta.append(eta_fun(socl, soch, 2500))
            eta_init_index = k

    soch, socl = state_leaving_matrix[1, eta_init_index], state_reaching_matrix[1, -1]
    eta.append(eta_fun(socl, soch, 2500))


@dataclass
class ElectricVehicle:
    # Mandatory attributes
    id: int
    weight: Union[int, float]
    battery_capacity: Union[int, float]
    alpha_up: Union[int, float]
    alpha_down: Union[int, float]
    max_tour_duration: Union[int, float]
    max_payload: Union[int, float]

    # Dynamic-related variables
    assigned_customers: Tuple[int, ...] = None
    x1_0: float = 0.0
    x2_0: float = 100.0
    x3_0: float = 0.0
    eta0: float = 1.0
    route: RouteVector = None
    travel_times: ndarray = None
    energy_consumption: ndarray = None
    charging_times: ndarray = None
    state_reaching: ndarray = None
    state_leaving: ndarray = None
    eta: List[float] = None

    def set_customers_to_visit(self, new_customers: Tuple[int, ...]):
        self.assigned_customers = new_customers

    def assign_customers_in_route(self, network: Network):
        self.assigned_customers = tuple([node for node in self.route[0] if network.isCustomer(node)])

    def set_route(self, route: RouteVector, x1_0: float, x2_0: float, x3_0: float, stochastic=False):
        self.route = route
        self.x1_0 = x1_0
        self.x2_0 = x2_0
        self.x3_0 = x3_0
        len_route = len(route[0])

        if stochastic:
            # TODO define sizes properly for the stochastic case
            size_state_matrix = (3, len_route)
            size_tt = (1, len_route - 1)
            size_ec = (1, len_route - 1)
            size_c_op = (1, len_route)
            init_state = (self.x1_0, self.x2_0, self.x3_0)
        else:
            size_state_matrix = (3, len_route)
            size_tt = (1, len_route - 1)
            size_ec = (1, len_route - 1)
            size_c_op = (1, len_route)
            init_state = (self.x1_0, self.x2_0, self.x3_0)

        # Matrices
        self.state_reaching = np.zeros(size_state_matrix)
        self.state_leaving = np.zeros(size_state_matrix)

        # Initial conditions
        self.state_reaching[:, 0] = np.array(init_state)
        self.state_leaving[:, 0] = np.array(init_state)
        self.state_leaving[2, -1] = self.weight

        # Other variables
        self.travel_times = np.zeros(size_tt)
        self.energy_consumption = np.zeros(size_ec)
        self.charging_times = np.zeros(size_c_op)
        self.charging_times[0, 0] = route[1][0]
        self.eta = [self.eta0]

    def iterate_space(self, network: Network):
        # k = len(self.route[0]) - 1
        # F_recursive(k, self.route, self.state_reaching, self.state_leaving, self.travel_times, self.energy_consumption,
        #            self.charging_times, self.weight, network)
        F_step(self.route, self.state_reaching, self.state_leaving, self.travel_times, self.energy_consumption,
               self.charging_times, self.weight, self.eta, network)

    def xml_element(self, assign_customers=False, with_routes=False):
        attribs = {'id': str(self.id),
                   'weight': str(self.weight),
                   'battery_capacity': str(self.battery_capacity),
                   'alpha_up': str(self.alpha_up),
                   'alpha_down': str(self.alpha_down),
                   'max_tour_duration': str(self.max_tour_duration),
                   'max_payload': str(self.max_payload)}
        element = ET.Element('electric_vehicle', attrib=attribs)

        if assign_customers:
            _assigned_customers = ET.SubElement(element, 'assigned_customers')
            for customer in self.assigned_customers:
                _customer = ET.SubElement(_assigned_customers, 'node', attrib={'id': str(customer)})

        if with_routes:
            _previous_route = ET.SubElement(element, 'previous_route')
            for Sk, Lk in self.route[0], self.route[1]:
                _point = ET.SubElement(_previous_route, 'node', attrib={'Sk': str(Sk), 'Lk': str(Lk)})

        return element
