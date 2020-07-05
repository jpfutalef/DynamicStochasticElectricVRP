from dataclasses import dataclass
from typing import Union, Tuple, Dict, NamedTuple, List

import numpy as np
from sklearn.neighbors import NearestNeighbors
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


def saturate(val, min_val, max_val):
    if val > max_val:
        return max_val
    elif val < min_val:
        return min_val
    return val


def F_step(route: RouteVector, state_reaching_matrix: ndarray, state_leaving_matrix: ndarray,
           tt_array: ndarray, ec_array: ndarray, c_op_array: ndarray, ev_weight: float, eta: List[float],
           network: Network, eta_table: np.ndarray, eta_model: NearestNeighbors):
    Sk, Lk = route[0], route[1]
    eta_init_index = 0
    for k, (Sk0, Lk0, Sk1, Lk1) in enumerate(zip(Sk[:-1], Lk[:-1], Sk[1:], Lk[1:]), 1):
        # TODO add waiting time
        departure_time = state_leaving_matrix[0, k - 1]
        payload = state_leaving_matrix[2, k - 1]

        tij = network.t(Sk0, Sk1, departure_time)
        eij = network.e(Sk0, Sk1, payload, ev_weight, departure_time) / eta[-1]
        tt_array[0, k - 1] = tij
        ec_array[0, k - 1] = eij

        state_reaching_matrix[:, k] = state_leaving_matrix[:, k - 1] + np.array([tij, -eij, 0])

        if network.isChargingStation(Sk1):
            soch = saturate(state_leaving_matrix[1, eta_init_index], 0., 100.)
            socl = saturate(state_reaching_matrix[1, k], 0., 100.)
            eta.append(eta[-1] * eta_fun(socl, soch, 2000, eta_table, eta_model))
            eta_init_index = k

        tj = network.spent_time(Sk1, state_reaching_matrix[1, k], Lk1, eta[-1])
        dj = network.demand(Sk1)

        state_leaving_matrix[:, k] = state_reaching_matrix[:, k] + np.array([tj, Lk1, -dj])

        if network.isChargingStation(Sk1):
            c_op_array[0, k] = tj

    soch = saturate(state_leaving_matrix[1, eta_init_index], 0., 100.)
    socl = saturate(state_reaching_matrix[1, -1], 0., 100.)
    eta.append(eta[-1] * eta_fun(socl, soch, 2000, eta_table, eta_model))


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

    def iterate_space(self, network: Network, eta_table: np.ndarray, eta_model: NearestNeighbors):
        F_step(self.route, self.state_reaching, self.state_leaving, self.travel_times, self.energy_consumption,
               self.charging_times, self.weight, self.eta, network, eta_table, eta_model)
        '''
        Sk, Lk = self.route[0], self.route[1]
        for k, (Sk0, Lk0, Sk1, Lk1) in enumerate(zip(Sk[:-1], Lk[:-1], Sk[1:], Lk[1:]), 1):
            # TODO add waiting time
            departure_time = self.state_leaving[0, k - 1]
            payload = self.state_leaving[2, k - 1]

            tij = network.t(Sk0, Sk1, departure_time)
            eij = network.e(Sk0, Sk1, payload, self.weight, departure_time) / self.eta0
            self.travel_times[0, k - 1] = tij
            self.energy_consumption[0, k - 1] = eij

            self.state_reaching[:, k] = self.state_reaching[:, k - 1] + np.array([tij, -eij, 0])

            tj = network.spent_time(Sk1, self.state_reaching[1, k], Lk1)
            dj = network.demand(Sk1)

            self.state_leaving[:, k] = self.state_reaching[:, k] + np.array([tj, Lk1, -dj])

            if network.isChargingStation(Sk1):
                self.charging_times[0, k] = tj

        socl, soch = min(self.state_reaching[1, :]), max(self.state_leaving[1, :])
        self.eta = eta_fun(socl, soch, 2000)
        '''

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
