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
        self.charging_times[0, 0] = route[1][0]  # TODO what's this?
        self.eta = [self.eta0]

    def step(self, network: Network):
        Sk, Lk = self.route[0], self.route[1]
        for k, (Sk0, Lk0, Sk1, Lk1) in enumerate(zip(Sk[:-1], Lk[:-1], Sk[1:], Lk[1:]), 1):
            # TODO add waiting time
            departure_time = self.state_leaving[0, k - 1]
            payload = self.state_leaving[2, k - 1]

            self.travel_times[0, k - 1] = tij = network.t(Sk0, Sk1, departure_time)
            self.energy_consumption[0, k - 1] = eij = network.e(Sk0, Sk1, payload, self.weight, departure_time)

            self.state_reaching[:, k] = self.state_leaving[:, k - 1] + np.array([tij, -eij, 0])

            tj = network.spent_time(Sk1, self.state_reaching[1, k], Lk1)
            dj = network.demand(Sk1)

            self.state_leaving[:, k] = self.state_reaching[:, k] + np.array([tj, Lk1, -dj])

            if network.isChargingStation(Sk1):
                self.charging_times[0, k] = tj

    def step_degradation(self, network: Network, eta_table: np.ndarray = None, eta_model: NearestNeighbors = None):
        Sk, Lk = self.route[0], self.route[1]
        eta_init_index = 0
        for k, (Sk0, Lk0, Sk1, Lk1) in enumerate(zip(Sk[:-1], Lk[:-1], Sk[1:], Lk[1:]), 1):
            # TODO add waiting time
            departure_time = self.state_leaving[0, k - 1]
            payload = self.state_leaving[2, k - 1]

            self.travel_times[0, k - 1] = tij = network.t(Sk0, Sk1, departure_time)
            self.energy_consumption[0, k - 1] = eij = network.e(Sk0, Sk1, payload, self.weight,
                                                                departure_time) / self.eta[-1]

            self.state_reaching[:, k] = self.state_leaving[:, k - 1] + np.array([tij, -eij, 0])

            if network.isChargingStation(Sk1):
                soch = saturate(self.state_leaving[1, eta_init_index], 0., 100.)
                socl = saturate(self.state_reaching[1, k], 0., 100.)
                self.eta.append(self.eta[-1] * eta_fun(socl, soch, 2000, eta_table, eta_model))
                eta_init_index = k

            tj = network.spent_time(Sk1, self.state_reaching[1, k], Lk1, self.eta[-1])
            dj = network.demand(Sk1)

            self.state_leaving[:, k] = self.state_reaching[:, k] + np.array([tj, Lk1, -dj])

            if network.isChargingStation(Sk1):
                self.charging_times[0, k] = tj

        soch = saturate(self.state_leaving[1, eta_init_index], 0., 100.)
        socl = saturate(self.state_reaching[1, -1], 0., 100.)
        self.eta.append(self.eta[-1] * eta_fun(socl, soch, 2000, eta_table, eta_model))

    def xml_element(self, assign_customers=False, with_routes=False, this_id=None):
        the_id = this_id if this_id is not None else self.id
        attribs = {'id': str(the_id),
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
            for Sk, Lk in zip(self.route[0], self.route[1]):
                _point = ET.SubElement(_previous_route, 'node', attrib={'Sk': str(Sk), 'Lk': str(Lk)})

        return element
