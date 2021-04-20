from dataclasses import dataclass
from typing import Union, Tuple, Dict, NamedTuple, List

import numpy as np
from sklearn.neighbors import NearestNeighbors
from numpy import ndarray, array
import xml.etree.ElementTree as ET

from res.models.Network import Network
from res.models.BatteryDegradation import eta as eta_fun

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
    current_max_tour_duration: Union[int, float] = 0.
    assigned_customers: Tuple[int, ...] = ()
    x1_0: float = 0.0
    x2_0: float = 100.0
    x3_0: float = 0.0
    route: RouteVector = None
    travel_times: ndarray = None
    energy_consumption: ndarray = None
    charging_times: ndarray = None
    waiting_times: ndarray = None
    waiting_times0: ndarray = None
    waiting_times1: ndarray = None
    service_time: ndarray = None
    state_reaching: ndarray = None
    state_leaving: ndarray = None
    state_reaching_covariance: ndarray = None
    state_leaving_covariance: ndarray = None
    with_state_reaching: bool = False

    def __post_init__(self):
        self.current_max_tour_duration = self.max_tour_duration

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
        self.waiting_times0 = None
        self.waiting_times1 = None
        self.service_time = None
        self.state_reaching = None
        self.state_leaving = None
        self.state_reaching_covariance = None
        self.state_leaving_covariance = None
        self.with_state_reaching = False
        self.current_max_tour_duration = self.max_tour_duration

    def set_customers_to_visit(self, new_customers: Tuple[int, ...]):
        self.assigned_customers = new_customers

    def assign_customers_in_route(self, network: Network):
        self.assigned_customers = tuple([node for node in self.route[0] if network.isCustomer(node)])

    def set_route(self, route: RouteVector, x1_0: float, x2_0: float, x3_0: float, init_covariance: np.array = None):
        self.route = route
        self.x1_0 = x1_0
        self.x2_0 = x2_0
        self.x3_0 = x3_0
        len_route = len(route[0])

        size_state_matrix = (3, len_route)
        size_state_covariance_matrix = (3, 3*len_route)
        size_tt = len_route - 1
        size_ec = len_route - 1
        size_wt = len_route
        size_c_op = len_route
        init_state = (self.x1_0, self.x2_0, self.x3_0)

        # Matrices
        self.state_reaching = np.zeros(size_state_matrix)
        self.state_leaving = np.zeros(size_state_matrix)
        self.state_reaching_covariance = np.zeros(size_state_covariance_matrix)
        self.state_leaving_covariance = np.zeros(size_state_covariance_matrix)

        # Initial conditions
        self.state_reaching[:, 0] = np.array(init_state)
        if init_covariance:
            self.state_reaching_covariance[0:3, 0:3] = init_covariance

        # Other variables
        self.travel_times = np.zeros(size_tt)
        self.energy_consumption = np.zeros(size_ec)
        self.waiting_times = np.zeros(size_wt)
        self.waiting_times0 = np.zeros(size_wt)
        self.waiting_times1 = np.zeros(size_wt)
        self.charging_times = np.zeros(size_c_op)
        self.service_time = np.zeros(size_wt)

    def step(self, network: Network):
        S, L = self.route[0], self.route[1]
        Sk0, Sk1, Lk0, Lk1, k = S[0], S[1], L[0], L[1], 0
        for k, (Sk0, Sk1, Lk0, Lk1) in enumerate(zip(S[:-1], S[1:], L[:-1], L[1:])):
            spent_time = network.spent_time(Sk0, self.state_reaching[1, k], Lk0)
            f0k = np.array([spent_time, Lk0, -network.demand(Sk0)])
            self.state_leaving[:, k] = self.state_reaching[:, k] + f0k

            (t, sigma_t), (e, sigma_e) = network.arc_costs(Sk0, Sk1, self.state_leaving[2, k], self.weight,
                                                           self.state_leaving[0, k])

            self.travel_times[k] = t
            self.energy_consumption[k] = e
            self.charging_times[k] = spent_time if network.isChargingStation(Sk0) else 0.
            self.service_time[k] = spent_time

            f1k = np.array([t, -e, 0])
            self.state_reaching[:, k+1] = self.state_leaving[:, k] + f1k

            Q = np.array([[sigma_t ** 2, sigma_t * sigma_e, 0], [sigma_t * sigma_e, sigma_e ** 2, 0], [0, 0, 0]])
            self.state_reaching_covariance[:, 3 * (k + 1):3 * (k + 1) + 3] = self.state_reaching_covariance[:, 3 * k:3 * k + 3] + Q

        # Update what happens at the end
        spent_time = network.spent_time(Sk1, self.state_reaching[1, k], Lk1)
        f0k = np.array([spent_time, Lk1, -network.demand(Sk1)])
        self.service_time[k+1] = spent_time
        self.state_leaving[:, k+1] = self.state_reaching[:, k+1] + f0k

    def xml_element(self, assign_customers=False, with_routes=False, this_id=None, online=False):
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
            _previous_route = ET.SubElement(element, 'previous_route', attrib={'x1': str(self.x1_0),
                                                                               'x2': str(self.x2_0),
                                                                               'x3': str(self.x3_0)})
            for Sk, Lk in zip(self.route[0], self.route[1]):
                _point = ET.SubElement(_previous_route, 'node', attrib={'Sk': str(Sk), 'Lk': str(Lk)})
        if online:
            _previous_route = ET.SubElement(element, 'online_route', attrib={'x1': str(self.x1_0),
                                                                             'x2': str(self.x2_0),
                                                                             'x3': str(self.x3_0)})
            for Sk, Lk in zip(self.route[0], self.route[1]):
                _point = ET.SubElement(_previous_route, 'node', attrib={'Sk': str(Sk), 'Lk': str(Lk)})
        return element
