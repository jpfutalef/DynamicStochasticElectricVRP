import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Union, Tuple, NamedTuple, List
import numpy as np

import res.models.Network as Network
from res.models.Penalization import penalization_deterministic, penalization_stochastic, normal_cdf


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


"""
ELECTRIC VEHICLE CLASS
"""


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

    # Vehicle properties
    Cr = 1.75
    c1 = 4.575
    c2 = 1.75
    rho_air = 1.2256
    Af = 2.3316
    Cd = 0.28
    type: str = None

    # Evaluation-related variables
    current_max_tour_duration: Union[int, float] = None
    assigned_customers: Tuple[int, ...] = ()
    x1_0: Union[float, None] = None
    x2_0: Union[float, None] = None
    x3_0: Union[float, None] = None
    S: Union[Tuple[int, ...], None] = None
    L: Union[Tuple[float, ...], None] = None
    travel_times: Union[np.ndarray, None] = None
    energy_consumption: Union[np.ndarray, None] = None
    charging_times: Union[np.ndarray, None] = None
    service_time: Union[np.ndarray, None] = None
    state_reaching: Union[np.ndarray, None] = None
    state_leaving: Union[np.ndarray, None] = None
    penalization: float = 0.0

    def __post_init__(self):
        if self.current_max_tour_duration is None:
            self.current_max_tour_duration = self.max_tour_duration
        self.type = self.__class__.__name__

    def reset(self, network: Network.Network):
        self.assigned_customers = ()
        self.x1_0 = None
        self.x2_0 = None
        self.x3_0 = None
        self.S = None
        self.L = None
        self.travel_times = None
        self.energy_consumption = None
        self.charging_times = None
        self.service_time = None
        self.state_reaching = None
        self.state_leaving = None

    def set_customers_to_visit(self, new_customers: Tuple[int, ...]):
        self.assigned_customers = new_customers

    def assign_customers_in_route(self, network: Network.Network):
        self.assigned_customers = tuple([node for node in self.S if network.is_customer(node)])

    def set_route(self, S: Tuple[int, ...], L: Tuple[float, ...], x1_0: float, x2_0: float, x3_0: float):
        len_route = len(S)
        init_state = np.array((x1_0, x2_0, x3_0))

        self.S = S
        self.L = L
        self.x1_0 = x1_0
        self.x2_0 = x2_0
        self.x3_0 = x3_0

        # Matrices
        self.state_reaching = np.zeros((3, len_route))
        self.state_leaving = np.zeros((3, len_route))

        # Initial conditions
        self.state_reaching[:, 0] = init_state

        # Variables containers
        self.travel_times = np.zeros(len_route - 1)
        self.energy_consumption = np.zeros(len_route - 1)
        self.charging_times = np.zeros(len_route)
        self.service_time = np.zeros(len_route)

    def step(self, network: Network.DeterministicCapacitatedNetwork, g=9.8):
        S, L = self.S, self.L
        Sk0, Sk1, Lk0, Lk1, k = S[0], S[1], L[0], L[1], 0
        for k, (Sk0, Sk1, Lk0, Lk1) in enumerate(zip(S[:-1], S[1:], L[:-1], L[1:])):
            spent_time = network.spent_time(Sk0, self.state_reaching[1, k], Lk0)
            f0k = np.array([spent_time, Lk0, -network.demand(Sk0)])
            self.state_leaving[:, k] = self.state_reaching[:, k] + f0k

            v_ij = network.v(Sk0, Sk1, self.state_leaving[0, k])  # m/s
            v_ij = v_ij[0] if len(v_ij) > 1 else v_ij

            edge = network.edges[Sk0][Sk1]
            m = self.weight + self.state_leaving[2, k]  # kg
            F = edge.road_cos_length * g * self.Cr * 1e-3 * (self.c1 * v_ij + self.c2) + edge.road_sin_length * g
            G = self.rho_air * self.Af * self.Cd * v_ij ** 2 * edge.length / 2.

            E_ij = m * F + G  # J
            e_ij = 100 * E_ij / self.battery_capacity
            t_ij = edge.length / v_ij if v_ij else 0.0  # s

            self.travel_times[k] = t_ij
            self.energy_consumption[k] = e_ij
            self.charging_times[k] = spent_time if network.is_charging_station(Sk0) else 0.
            self.service_time[k] = spent_time

            f1k = np.array([t_ij, -e_ij, 0])
            self.state_reaching[:, k + 1] = self.state_leaving[:, k] + f1k

        # Update what happens at the end
        spent_time = network.spent_time(Sk1, self.state_reaching[1, k], Lk1)
        f0k = np.array([spent_time, Lk1, -network.demand(Sk1)])
        self.service_time[k + 1] = spent_time
        self.state_leaving[:, k + 1] = self.state_reaching[:, k + 1] + f0k

    def check_feasibility(self, network: Network.Network, penalization_constant=500000.0):
        self.penalization = 0.0
        self.constraint_max_tour_time()
        self.constraint_max_payload()
        self.constraint_time_window_and_soc_policy(network)
        if self.penalization > 0.:
            self.penalization += penalization_constant
        return self.penalization

    def constraint_max_tour_time(self):
        if self.state_reaching[0, -1] - self.state_reaching[0, 0] > self.current_max_tour_duration:
            self.penalization += penalization_deterministic(self.max_tour_duration,
                                                            self.state_reaching[0, -1] - self.state_leaving[0, 0])

    def constraint_max_payload(self):
        if self.state_leaving[2, 0] > self.max_payload:
            self.penalization += penalization_deterministic(self.state_leaving[2, 0], self.max_payload)

    def constraint_time_window_and_soc_policy(self, network: Network.Network):
        for k, (Sk, Lk) in enumerate(zip(self.S, self.L)):
            node = network.nodes[Sk]
            if node.is_customer():
                # TIME WINDOW LOW BOUND
                if node.time_window_low > self.state_reaching[0, k]:
                    self.penalization += penalization_deterministic(self.state_reaching[0, k], node.time_window_low)

                # TIME WINDOW UPPER BOUND
                if node.time_window_upp < self.state_leaving[0, k]:
                    self.penalization += penalization_deterministic(node.time_window_upp, self.state_leaving[0, k])

            # SOC BOUND LOWER - REACHING
            if self.state_reaching[1, k] < self.alpha_down:
                self.penalization += penalization_deterministic(self.state_reaching[1, k], self.alpha_down)

            # SOC BOUND UPPER - REACHING
            if self.state_reaching[1, k] > self.alpha_up:
                self.penalization += penalization_deterministic(self.state_reaching[1, k], self.alpha_up)

            # SOC BOUND LOWER - LEAVING
            if self.state_leaving[1, k] < self.alpha_down:
                self.penalization += penalization_deterministic(self.state_leaving[1, k], self.alpha_down)

            # SOC BOUND UPPER - LEAVING
            if self.state_leaving[1, k] > self.alpha_up:
                self.penalization += penalization_deterministic(self.state_leaving[1, k], self.alpha_up)

            # SOC 0 - REACHING
            if self.state_reaching[1, k] < 0:
                self.penalization += penalization_deterministic(self.state_reaching[1, k], 0, c=1e5, w=1e3)

            # SOC 100 - REACHING
            if self.state_reaching[1, k] > 100:
                self.penalization += penalization_deterministic(self.state_reaching[1, k], 100, c=1e5, w=1e3)

            # SOC 0 - LEAVING
            if self.state_leaving[1, k] < 0:
                self.penalization += penalization_deterministic(self.state_leaving[1, k], 0, c=1e5, w=1e3)

            # SOC 100 - LEAVING
            if self.state_leaving[1, k] > 100:
                self.penalization += penalization_deterministic(self.state_leaving[1, k], 100, c=1e5, w=1e3)

    def plot_operation(self, ax):
        return

    def xml_element(self, assign_customers=False, with_route=False, this_id=None):
        the_id = this_id if this_id is not None else self.id
        attribs = {'id': str(the_id),
                   'weight': str(self.weight),
                   'battery_capacity': str(self.battery_capacity),
                   'battery_capacity_nominal': str(self.battery_capacity_nominal),
                   'alpha_up': str(self.alpha_up),
                   'alpha_down': str(self.alpha_down),
                   'max_tour_duration': str(self.max_tour_duration),
                   'max_payload': str(self.max_payload),
                   'type': self.type}
        element = ET.Element('electric_vehicle', attrib=attribs)

        if assign_customers:
            _assigned_customers = ET.SubElement(element, 'assigned_customers')
            [_assigned_customers.append(ET.Element('node', attrib={'id': str(i)})) for i in self.assigned_customers]

        if with_route:
            _route = ET.SubElement(element, 'route', attrib={'x1': str(self.x1_0), 'x2': str(self.x2_0),
                                                             'x3': str(self.x3_0)})
            [_route.append(ET.Element('node', attrib={'Sk': str(Sk), 'Lk': str(Lk)})) for Sk, Lk in zip(self.S, self.L)]
        return element

    @classmethod
    def from_xml_element(cls, element: ET.Element):
        ev_id = int(element.get('id'))
        weight = float(element.get('weight'))
        max_tour_duration = float(element.get('max_tour_duration'))
        max_payload = float(element.get('max_payload'))
        battery_capacity_nominal = float(element.get('battery_capacity_nominal'))
        batter_capacity = float(element.get('battery_capacity'))
        alpha_up = float(element.get('alpha_up'))
        alpha_down = float(element.get('alpha_down'))
        _assigned_customers = element.find('assigned_customers')
        if _assigned_customers:
            assigned_customers = tuple([int(i.get('id')) for i in _assigned_customers])
        else:
            assigned_customers = None
        return cls(ev_id, weight, batter_capacity, battery_capacity_nominal, alpha_up, alpha_down, max_tour_duration,
                   max_payload, assigned_customers=assigned_customers)


"""
GAUSSIAN ELECTRIC VEHICLE CLASS
"""


def probability_in_node(probability_container: np.ndarray, sample_time: float, mu: float, sigma: float,
                        spent_time: float):
    """
    Calculates the probability an EV is a node using Gaussian statistics
    @param probability_container: an array that will store the probabilities for each TOD
    @param sample_time: time resolution of the calculation
    @param mu: expected time the EV arrives at the node
    @param sigma: std of the time the EV arrives at the node
    @param spent_time: the service time at the node
    @return: None. The probability container is modified in place to store the probability values
    """
    t = 0
    for i, _ in enumerate(probability_container):
        cdf1 = normal_cdf(t, mu, sigma)
        cdf2 = 1 - normal_cdf(t, mu + spent_time, sigma)
        probability_container[i] = cdf1 * cdf2
        t += sample_time


@dataclass
class GaussianElectricVehicle(ElectricVehicle):
    state_reaching_covariance: Union[np.ndarray, None] = None
    state_leaving_covariance: Union[np.ndarray, None] = None
    Q: Union[np.ndarray, None] = None
    visits_a_cs: bool = False
    visited_nodes: Union[np.ndarray, None] = None
    probability_in_cs: Union[np.ndarray, None] = None
    sample_time_probability_in_cs: float = 5.0
    num_customers: int = 0

    def __post_init__(self):
        super(GaussianElectricVehicle, self).__post_init__()
        self.Q = np.zeros((3, 3))

    def reset(self, network: Network.Network):
        super(GaussianElectricVehicle, self).reset(network)
        self.state_reaching_covariance = None
        self.state_leaving_covariance = None
        self.reset_visited_nodes_array(network)

    def create_visited_nodes_array(self, network: Network.GaussianCapacitatedNetwork):
        self.visited_nodes = np.zeros(len(network))

    def create_probability_in_cs_array(self, network_length: int):
        self.probability_in_cs = np.zeros(network_length, 24 * 60 / self.sample_time_probability_in_cs)

    def reset_visited_nodes_array(self, network: Network.Network):
        if self.visited_nodes is None:
            self.create_visited_nodes_array(network)
        self.visited_nodes.fill(0)

    def reset_probability_in_cs_array(self):
        self.probability_in_cs.fill(0)

    def set_num_of_customers(self, num_customers: int):
        self.num_customers = num_customers

    def set_route(self, S: Tuple[int, ...], L: Tuple[float, ...], x1_0: float, x2_0: float, x3_0: float,
                  init_covariance: np.array = None):
        super(GaussianElectricVehicle, self).set_route(S, L, x1_0, x2_0, x3_0)
        self.state_reaching_covariance = np.zeros((3, 3 * len(self.S)))
        if init_covariance:
            self.state_reaching_covariance[0:3, 0:3] = init_covariance

    def step(self, network: Network.GaussianCapacitatedNetwork, g=9.8):
        S, L = self.S, self.L
        Sk0, Sk1, Lk0, Lk1, k = S[0], S[1], L[0], L[1], 0
        for k, (Sk0, Sk1, Lk0, Lk1) in enumerate(zip(S[:-1], S[1:], L[:-1], L[1:])):
            spent_time = network.spent_time(Sk0, self.state_reaching[1, k], Lk0)
            f0k = np.array([spent_time, Lk0, -network.demand(Sk0)])
            self.state_leaving[:, k] = self.state_reaching[:, k] + f0k

            edge = network.edges[Sk0][Sk1]
            v_ij, sigma_ij = network.v(Sk0, Sk1, self.state_leaving[0, k])  # m/s
            m = self.weight + self.state_leaving[2, k]  # kg

            a1 = self.rho_air * self.Af * self.Cd * edge.length * v_ij ** 2 / 2
            a2 = edge.road_cos_length * g * self.Cr * self.c1 * m * v_ij / 1000
            a3 = edge.road_sin_length * g * m
            b1 = self.rho_air * self.Af * self.Cd * edge.length * v_ij
            b2 = edge.road_cos_length * g * self.Cr * self.c1 * m / 1000

            mu_E_ij = a1 + a2 + a3  # J
            sigma_E_ij = (b1 + b2) * sigma_ij   # J

            mu_e_ij = 100 * mu_E_ij / self.battery_capacity
            sigma_e_ij = 100 * sigma_E_ij / self.battery_capacity

            mu_t_ij = edge.length / v_ij if v_ij else 0.0  # s
            sigma_t_ij = -edge.length * sigma_ij / v_ij ** 2  if v_ij else 0.0  # s

            f1k = np.array([mu_t_ij, -mu_e_ij, 0])
            self.state_reaching[:, k + 1] = self.state_leaving[:, k] + f1k

            self.Q[0, 0] = sigma_t_ij ** 2
            self.Q[0, 1] = sigma_t_ij * sigma_e_ij
            self.Q[0, 2] = 0
            self.Q[1, 0] = sigma_t_ij * sigma_e_ij
            self.Q[1, 1] = sigma_e_ij ** 2
            self.Q[1, 2] = 0
            self.Q[2, 0] = 0
            self.Q[2, 1] = 0
            self.Q[2, 2] = 0
            self.state_reaching_covariance[:, 3 * (k + 1):3 * (k + 1) + 3] = self.state_reaching_covariance[:,
                                                                             3 * k:3 * k + 3] + self.Q

            self.travel_times[k] = mu_t_ij
            self.energy_consumption[k] = mu_e_ij
            self.charging_times[k] = spent_time if network.is_charging_station(Sk0) else 0.
            self.service_time[k] = spent_time

            self.visited_nodes[Sk0] += 1
            if network.is_charging_station(Sk0):
                self.visits_a_cs = True

        # Update what happens at the end (leaving)
        spent_time = network.spent_time(Sk1, self.state_reaching[1, k], Lk1)
        f0k = np.array([spent_time, Lk1, -network.demand(Sk1)])
        self.service_time[k + 1] = spent_time
        self.state_leaving[:, k + 1] = self.state_reaching[:, k + 1] + f0k
        self.visited_nodes[Sk1] += 1

    def probability_in_node(self, k: int):
        Sk = self.S[k]
        sample_time = self.sample_time_probability_in_cs
        mu = self.state_reaching[0, k]
        sigma = np.sqrt(self.state_reaching_covariance[0, 3 * k])  # TODO is this better than math.sqrt(.)?
        spent_time = self.service_time[k]
        probability_in_node(self.probability_in_cs[Sk, :], sample_time, mu, sigma, spent_time)

    def check_feasibility(self, network: Network.Network, penalization_constant=0.) -> float:
        self.penalization = 0.0
        self.constraint_max_tour_time()
        self.constraint_max_payload()
        self.constraint_time_window_and_soc_policy(network)
        if self.penalization > 0.:
            self.penalization += penalization_constant
        return self.penalization

    def constraint_max_tour_time(self, PRB=.9545):
        MU = self.state_reaching[0, -1]
        SIG = np.sqrt(self.state_reaching_covariance[0, -3])
        VAL = self.max_tour_duration + self.x1_0
        CDF = normal_cdf(VAL, MU, SIG)
        self.penalization += penalization_stochastic(CDF, PRB, w=1e4)

    def constraint_max_payload(self):
        if self.state_leaving[2, 0] > self.max_payload:
            self.penalization += penalization_deterministic(self.state_leaving[2, 0], self.max_payload)

    def constraint_time_window_and_soc_policy(self, network: Network.Network):
        for k, (Sk, Lk) in enumerate(zip(self.S, self.L)):
            node = network.nodes[Sk]
            if node.is_customer():
                """
                TIME WINDOW - LOWER BOUND
                """
                PRB = .9545
                MU = self.state_reaching[0, k]
                SIG = np.sqrt(self.state_reaching_covariance[0, 3 * k])
                VAL = node.time_window_low
                CDF = 1 - normal_cdf(VAL, MU, SIG)
                self.penalization += penalization_stochastic(CDF, PRB, w=1e4)

                """
                TIME WINDOW - UPPER BOUND
                """
                PRB = .9545
                MU = self.state_leaving[0, k]
                SIG = np.sqrt(self.state_reaching_covariance[0, 3 * k])
                VAL = node.time_window_upp
                CDF = normal_cdf(VAL, MU, SIG)
                self.penalization += penalization_stochastic(CDF, PRB, w=1e4)

            """
            SOC BOUND - REACHING
            """
            PRB = .9545
            MU = self.state_reaching[1, k]
            SIG = np.sqrt(self.state_reaching_covariance[1, 3 * k + 1])
            VAL1 = self.alpha_up
            VAL2 = self.alpha_down
            CDF1 = normal_cdf(VAL1, MU, SIG)
            CDF2 = normal_cdf(VAL2, MU, SIG)
            CDF = CDF1 - CDF2
            self.penalization += penalization_stochastic(CDF, PRB, w=1e4)

            """
            SOC BOUND LOWER - LEAVING
            """
            PRB = .9545
            MU = self.state_leaving[1, k]
            SIG = SIG
            VAL1 = self.alpha_up
            VAL2 = self.alpha_down
            CDF1 = normal_cdf(VAL1, MU, SIG)
            CDF2 = normal_cdf(VAL2, MU, SIG)
            CDF = CDF1 - CDF2
            self.penalization += penalization_stochastic(CDF, PRB, w=1e4)

    def xml_element(self, assign_customers=False, with_route=False, this_id=None):
        element = super(GaussianElectricVehicle, self).xml_element(assign_customers, with_route, this_id)
        if with_route:
            _route = element.find('route')  # TODO add covariance matrix
        return element

    @classmethod
    def from_xml_element(cls, element: ET.Element):
        instance = super(GaussianElectricVehicle, cls).from_xml_element(element)
        return instance


def from_xml_element(element: ET.Element):
    t = element.get('type')
    cls = globals()[t]
    return cls.from_xml_element(element)
