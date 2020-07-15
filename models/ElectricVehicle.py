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
    battery_capacity_nominal: Union[int, float]
    alpha_up: Union[int, float]
    alpha_down: Union[int, float]
    max_tour_duration: Union[int, float]
    max_payload: Union[int, float]

    # Dynamic-related variables
    assigned_customers: Tuple[int, ...] = ()
    x1_0: float = 0.0
    x2_0: float = 100.0
    x3_0: float = 0.0
    route: RouteVector = None
    travel_times: ndarray = None
    energy_consumption: ndarray = None
    charging_times: ndarray = None
    waiting_times: ndarray = None
    state_reaching: ndarray = None
    state_leaving: ndarray = None

    def reset(self):
        self.battery_capacity = self.battery_capacity_nominal
        self.assigned_customers = ()
        self.x1_0 = 0.0
        self.x2_0 = 100.0
        self.x3_0 = 0.0
        self.route = None
        self.travel_times = None
        self.energy_consumption = None
        self.charging_times = None
        self.waiting_times = None
        self.state_reaching = None
        self.state_leaving = None

    def set_customers_to_visit(self, new_customers: Tuple[int, ...]):
        self.assigned_customers = new_customers

    def assign_customers_in_route(self, network: Network):
        self.assigned_customers = tuple([node for node in self.route[0] if network.isCustomer(node)])

    def set_route(self, route: RouteVector, x1_0: float, x2_0: float, x3_0: float):
        self.route = route
        self.x1_0 = x1_0
        self.x2_0 = x2_0
        self.x3_0 = x3_0
        len_route = len(route[0])

        size_state_matrix = (3, len_route)
        size_tt = (1, len_route - 1)
        size_ec = (1, len_route - 1)
        size_wt = (1, len_route)
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
        self.waiting_times = np.zeros(size_wt)
        self.charging_times = np.zeros(size_c_op)
        self.charging_times[0, 0] = route[1][0]  # TODO what's this?

    def step(self, network: Network):
        Sk, Lk = self.route[0], self.route[1]
        tij = network.t(Sk[0], Sk[1], self.state_leaving[0, 0])
        Eij = network.e(Sk[0], Sk[1], self.state_leaving[2, 0], self.weight, self.state_leaving[0, 0], tij*60)
        eij = Eij * 100 / self.battery_capacity
        for k, (Sk0, Lk0, Sk1, Lk1) in enumerate(zip(Sk[1:-1], Lk[1:-1], Sk[2:], Lk[2:]), 1):
            self.state_reaching[:, k] = self.state_leaving[:, k - 1] + np.array([tij, -eij, 0])
            self.travel_times[0, k - 1] = tij
            self.energy_consumption[0, k - 1] = eij

            eta = self.battery_capacity/self.battery_capacity_nominal
            ti = network.spent_time(Sk0, self.state_reaching[1, k], Lk0, eta)
            done_time = self.state_reaching[0, k] + ti
            di = network.demand(Sk0)
            payload_after = self.state_reaching[2, k] - di

            tij, Eij, wti = network.waiting_time(Sk0, Sk1, done_time, payload_after, self.weight)
            eij = Eij * 100 / self.battery_capacity

            self.state_leaving[:, k] = self.state_reaching[:, k] + np.array([ti + wti, Lk0, -di])

            self.waiting_times[0, k] = wti
            if network.isChargingStation(Sk0):
                self.charging_times[0, k] = ti

        self.travel_times[0, -1] = tij
        self.energy_consumption[0, -1] = eij
        self.state_reaching[:, -1] = self.state_leaving[:, -2] + np.array([tij, -eij, 0])
        self.state_leaving[:, -1] = self.state_reaching[:, -1]

    def step_degradation_eta(self, network: Network, eta_table: np.ndarray, eta_model: NearestNeighbors) -> List[float]:
        Sk, Lk = self.route[0], self.route[1]
        tij = network.t(Sk[0], Sk[1], self.state_leaving[0, 0])
        Eij = network.e(Sk[0], Sk[1], self.state_leaving[2, 0], self.weight, self.state_leaving[0, 0], tij*60)
        eij = Eij * 100 / self.battery_capacity
        deg_index, capacity_changes = 0, []
        for k, (Sk0, Lk0, Sk1, Lk1) in enumerate(zip(Sk[1:-1], Lk[1:-1], Sk[2:], Lk[2:]), 1):
            self.state_reaching[:, k] = self.state_leaving[:, k - 1] + np.array([tij, -eij, 0])
            self.travel_times[0, k - 1] = tij
            self.energy_consumption[0, k - 1] = eij

            if network.isChargingStation(Sk0):
                soch = saturate(self.state_leaving[1, deg_index], 0., 100.)
                socl = saturate(self.state_reaching[1, k], 0., 100.)
                self.battery_capacity *= eta_fun(socl, soch, 2000, eta_table, eta_model)
                capacity_changes.append(self.battery_capacity)
                deg_index = k

            eta = self.battery_capacity / self.battery_capacity_nominal
            ti = network.spent_time(Sk0, self.state_reaching[1, k], Lk0, eta)
            done_time = self.state_reaching[0, k] + ti
            di = network.demand(Sk0)
            payload_after = self.state_reaching[2, k] - di

            tij, Eij, wti = network.waiting_time(Sk0, Sk1, done_time, payload_after, self.weight)
            eij = Eij * 100 / self.battery_capacity

            self.state_leaving[:, k] = self.state_reaching[:, k] + np.array([ti + wti, Lk0, -di])

            self.waiting_times[0, k] = wti
            if network.isChargingStation(Sk0):
                self.charging_times[0, k] = ti

        self.travel_times[0, -1] = tij
        self.energy_consumption[0, -1] = eij
        self.state_reaching[:, -1] = self.state_leaving[:, -2] + np.array([tij, -eij, 0])
        self.state_leaving[:, -1] = self.state_reaching[:, -1]

        soch = saturate(self.state_leaving[1, deg_index], 0., 100.)
        socl = saturate(self.state_reaching[1, -1], 0., 100.)
        self.battery_capacity *= eta_fun(socl, soch, 2000, eta_table, eta_model)
        capacity_changes.append(self.battery_capacity)
        return capacity_changes

    def step_degradation_eta_capacity(self, network: Network, used_capacity: float, eta_table: np.ndarray,
                                      eta_model: NearestNeighbors) -> Tuple[List[float], float]:
        Sk, Lk = self.route[0], self.route[1]
        tij = network.t(Sk[0], Sk[1], self.state_leaving[0, 0])
        Eij = network.e(Sk[0], Sk[1], self.state_leaving[2, 0], self.weight, self.state_leaving[0, 0], tij*60)
        eij = Eij * 100 / self.battery_capacity
        capacity_changes = []
        for k, (Sk0, Lk0, Sk1, Lk1) in enumerate(zip(Sk[1:-1], Lk[1:-1], Sk[2:], Lk[2:]), 1):
            self.state_reaching[:, k] = self.state_leaving[:, k - 1] + np.array([tij, -eij, 0])
            self.travel_times[0, k - 1] = tij
            self.energy_consumption[0, k - 1] = eij

            used_capacity += Eij
            if used_capacity >= self.battery_capacity:
                used_capacity -= self.battery_capacity
                self.battery_capacity *= eta_fun(self.alpha_down, self.alpha_up, 2000, eta_table, eta_model)
                capacity_changes.append(self.battery_capacity)

            eta = self.battery_capacity / self.battery_capacity_nominal
            ti = network.spent_time(Sk0, self.state_reaching[1, k], Lk0, eta)
            done_time = self.state_reaching[0, k] + ti
            di = network.demand(Sk0)
            payload_after = self.state_reaching[2, k] - di

            tij, Eij, wti = network.waiting_time(Sk0, Sk1, done_time, payload_after, self.weight)
            eij = Eij * 100 / self.battery_capacity

            self.state_leaving[:, k] = self.state_reaching[:, k] + np.array([ti + wti, Lk0, -di])

            self.waiting_times[0, k] = wti
            if network.isChargingStation(Sk0):
                self.charging_times[0, k] = ti

        self.travel_times[0, -1] = tij
        self.energy_consumption[0, -1] = eij
        self.state_reaching[:, -1] = self.state_leaving[:, -2] + np.array([tij, -eij, 0])
        self.state_leaving[:, -1] = self.state_reaching[:, -1]
        return capacity_changes, used_capacity

    def xml_element(self, assign_customers=False, with_routes=False, this_id=None):
        the_id = this_id if this_id is not None else self.id
        attribs = {'id': str(the_id),
                   'weight': str(self.weight),
                   'battery_capacity': str(self.battery_capacity),
                   'battery_capacity_nominal': str(self.battery_capacity_nominal),
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
            _cp = ET.SubElement(element, 'critical_point', attrib={'x1': str(self.x1_0), 'x2': str(self.x2_0),
                                                                   'x3': str(self.x3_0), 'k': '0'})
        return element
