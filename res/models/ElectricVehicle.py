import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Union, Tuple, NamedTuple, List
import numpy as np
import matplotlib.pyplot as plt

import res.models.Network as Network
from res.models.Penalization import penalization_deterministic, penalization_stochastic, normal_cdf


class InitialCondition(NamedTuple):
    S0: int
    L0: float
    x1_0: float
    x2_0: float
    x3_0: float


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
    sos_state: Union[np.ndarray, None] = None
    eos_state: Union[np.ndarray, None] = None
    penalization: float = 0.0
    pre_service_waiting_time: Union[np.ndarray, None] = None
    post_service_waiting_time: Union[np.ndarray, None] = None

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
        self.sos_state = None
        self.eos_state = None
        self.pre_service_waiting_time = None
        self.post_service_waiting_time = None

    def set_customers_to_visit(self, new_customers: Tuple[int, ...]):
        self.assigned_customers = new_customers

    def assign_customers_in_route(self, network: Network.Network):
        self.assigned_customers = tuple([node for node in self.S if network.is_customer(node)])

    def set_route(self, S: Tuple[int, ...], L: Tuple[float, ...], x1_0: float, x2_0: float, x3_0: float,
                  first_pwt: float = 0.):
        len_route = len(S)
        init_state = np.array((x1_0, x2_0, x3_0))

        self.S = S
        self.L = L
        self.x1_0 = x1_0
        self.x2_0 = x2_0
        self.x3_0 = x3_0

        # Matrices
        self.sos_state = np.zeros((3, len_route))
        self.eos_state = np.zeros((3, len_route))

        # Initial conditions
        self.sos_state[:, 0] = init_state

        # Variables containers
        self.travel_times = np.zeros(len_route - 1)
        self.energy_consumption = np.zeros(len_route - 1)
        self.charging_times = np.zeros(len_route)
        self.service_time = np.zeros(len_route)
        self.pre_service_waiting_time = np.zeros(len_route)
        self.post_service_waiting_time = np.zeros(len_route)

    def tt(self, tod: float, edge: Network.Edge):
        v = edge.get_velocity(tod)[0]
        return travel_time(v, edge.length)

    def Ec(self, tod: float, m: float, g: float, edge: Network.Edge):
        v = edge.get_velocity(tod)[0]
        e = energy_consumption(v, edge.length, m, self.Cr, self.c1, self.c2, g, self.Af, self.Cd,
                               self.rho_air, edge.road_cos_length, edge.road_sin_length)
        return e

    def step(self, network: Network.DeterministicCapacitatedNetwork, g=9.8):
        S, L = self.S, self.L
        Sk0, Sk1, Lk0, Lk1, k = S[0], S[1], L[0], L[1], 0
        for k, (Sk0, Sk1, Lk0, Lk1) in enumerate(zip(S[:-1], S[1:], L[:-1], L[1:])):
            spent_time = network.spent_time(Sk0, self.sos_state[1, k], Lk0)
            f0k = np.array([spent_time, Lk0, -network.demand(Sk0)])
            self.eos_state[:, k] = self.sos_state[:, k] + f0k

            eos_time = self.eos_state[0, k]
            edge = network.edges[Sk0][Sk1]
            m = self.weight + self.eos_state[2, k]  # kg

            # v_ij, _ = network.v(Sk0, Sk1, eos_time)  # m/s
            # F = edge.road_cos_length * g * self.Cr * 1e-3 * (self.c1 * v_ij + self.c2) + edge.road_sin_length * g
            # G = self.rho_air * self.Af * self.Cd * v_ij ** 2 * edge.length / 2.
            # E_ij = m * F + G  # J
            # t_ij = edge.length / v_ij if v_ij else 0.0  # s

            t_ij = self.tt(eos_time, edge)
            E_ij = self.Ec(eos_time, m, g, edge)
            e_ij = 100 * E_ij / self.battery_capacity

            f1k = np.array([t_ij, -e_ij, 0])
            self.sos_state[:, k + 1] = self.eos_state[:, k] + f1k

            self.travel_times[k] = t_ij
            self.energy_consumption[k] = e_ij
            self.charging_times[k] = spent_time if network.is_charging_station(Sk0) else 0.
            self.service_time[k] = spent_time

        # Update what happens at the end
        k += 1
        spent_time = network.spent_time(self.S[k], self.sos_state[1, k], self.L[k])
        f0k = np.array([spent_time, self.L[k], -network.demand(self.S[k])])
        self.eos_state[:, k] = self.sos_state[:, k] + f0k

    def check_feasibility(self, network: Network.Network, penalization_constant=500000.0):
        self.penalization = 0.0
        self.constraint_max_tour_time()
        self.constraint_max_payload()
        self.constraint_time_window_and_soc_policy(network)
        if self.penalization > 0.:
            self.penalization += penalization_constant
        return self.penalization

    def constraint_max_tour_time(self):
        if self.sos_state[0, -1] - self.sos_state[0, 0] > self.current_max_tour_duration:
            self.penalization += penalization_deterministic(self.max_tour_duration,
                                                            self.sos_state[0, -1] - self.eos_state[0, 0])

    def constraint_max_payload(self):
        if self.eos_state[2, 0] > self.max_payload:
            self.penalization += penalization_deterministic(self.eos_state[2, 0], self.max_payload)

    def constraint_time_window_and_soc_policy(self, network: Network.Network):
        for k, (Sk, Lk) in enumerate(zip(self.S, self.L)):
            node = network.nodes[Sk]
            if node.is_customer():
                # TIME WINDOW LOW BOUND
                if node.time_window_low > self.sos_state[0, k]:
                    self.penalization += penalization_deterministic(self.sos_state[0, k], node.time_window_low)

                # TIME WINDOW UPPER BOUND
                if node.time_window_upp < self.eos_state[0, k]:
                    self.penalization += penalization_deterministic(node.time_window_upp, self.eos_state[0, k])

            # SOC BOUND LOWER - REACHING
            if self.sos_state[1, k] < self.alpha_down:
                self.penalization += penalization_deterministic(self.sos_state[1, k], self.alpha_down)

            # SOC BOUND UPPER - REACHING
            if self.sos_state[1, k] > self.alpha_up:
                self.penalization += penalization_deterministic(self.sos_state[1, k], self.alpha_up)

            # SOC BOUND LOWER - LEAVING
            if self.eos_state[1, k] < self.alpha_down:
                self.penalization += penalization_deterministic(self.eos_state[1, k], self.alpha_down)

            # SOC BOUND UPPER - LEAVING
            if self.eos_state[1, k] > self.alpha_up:
                self.penalization += penalization_deterministic(self.eos_state[1, k], self.alpha_up)

            # SOC 0 - REACHING
            if self.sos_state[1, k] < 0:
                self.penalization += penalization_deterministic(self.sos_state[1, k], 0, c=1e5, w=1e3)

            # SOC 100 - REACHING
            if self.sos_state[1, k] > 100:
                self.penalization += penalization_deterministic(self.sos_state[1, k], 100, c=1e5, w=1e3)

            # SOC 0 - LEAVING
            if self.eos_state[1, k] < 0:
                self.penalization += penalization_deterministic(self.eos_state[1, k], 0, c=1e5, w=1e3)

            # SOC 100 - LEAVING
            if self.eos_state[1, k] > 100:
                self.penalization += penalization_deterministic(self.eos_state[1, k], 100, c=1e5, w=1e3)

    def plot_operation(self, network: Network.Network, **kwargs):
        # Figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, **kwargs)

        # Figure 1
        self.draw_time_windows(network, ax1)
        self.draw_service_times(network, ax1)
        self.draw_travelling_times(ax1)
        self.draw_max_tour_time(ax1)
        self.draw_depots_times(network, ax1)

        ax1.set_xlabel('Stop')
        ax1.set_ylabel('Time of day [s]')
        ax1.set_title(f'Time EV {self.id}')
        ax1.grid()
        ax1.legend()

        # Figure 2
        self.draw_service_soc(network, ax2)
        self.draw_travelling_soc(ax2)
        self.draw_soc_policy(ax2)

        ax2.set_xlabel('Stop')
        ax2.set_ylabel('SOC [%]')
        ax2.set_title(f'SOC EV {self.id}')
        ax2.set_ylim((0, 100))
        ax2.set_xlim((-.5, len(self.S) - .5))
        ax2.grid()
        ax2.legend()

        # Figure 3
        self.draw_service_payload(ax3)
        self.draw_travelling_payload(ax3)
        ax3.axhline(self.max_payload, linestyle='--', color='black', label='Max. payload')

        ax3.set_xlabel('Stop')
        ax3.set_ylabel('Payload [kg]')
        ax3.set_title(f'Payload EV {self.id}')
        ax3.grid()
        ax3.legend()

        # Adjust figure
        fig.tight_layout()
        return fig, (ax1, ax2, ax3)

    def draw_time_windows(self, network: Network.Network, ax: plt.Axes):
        X = [k for k, _ in enumerate(self.S) if network.is_customer(self.S[k])]
        twl = [network.time_window_low(self.S[k]) for k in X]
        twu = [network.time_window_upp(self.S[k]) for k in X]
        twdiff = np.asarray(twu) - np.asarray(twl)
        twdescription = np.array([np.zeros_like(twl), twdiff])
        ax.errorbar(X, twl, twdescription, ecolor='black', fmt='none', capsize=6, elinewidth=1, )
        return

    def draw_service_times(self, network: Network.Network, ax: plt.Axes):
        X_customer = [k for k, _ in enumerate(self.S) if network.is_customer(self.S[k])]
        X_cs = [k for k, _ in enumerate(self.S) if network.is_charging_station(self.S[k])]

        # Arrows for customers
        draw_service_time_arrows(self, X_customer, ax, 'SeaGreen', 'Serving customer')

        # Arrows for CSs
        draw_service_time_arrows(self, X_cs, ax, 'Crimson', 'Recharging battery')
        return

    def draw_travelling_times(self, ax: plt.Axes):
        preX = list(range(len(self.S)))
        X = preX[:-1]
        Y = self.eos_state[0, :-1] + self.post_service_waiting_time[:-1]  # TODO Check
        U = np.ones_like(preX[1:])
        V = self.sos_state[0, 1:] - self.pre_service_waiting_time[1:] - Y
        ax.quiver(X, Y, U, V, scale=1, angles='xy', scale_units='xy', color='SteelBlue', width=0.0004 * 16, zorder=15,
                  label='Travelling')

    def draw_max_tour_time(self, ax: plt.Axes):
        ax.axhline(self.sos_state[0, 0] + self.current_max_tour_duration, linestyle='--', color='black',
                   label='Maximum tour time')
        return

    def draw_depots_times(self, network: Network.Network, ax: plt.Axes):
        X_depot = [k for k, _ in enumerate(self.S) if network.is_depot(self.S[k])]
        Y_depot = self.sos_state[0, X_depot]
        ax.plot(X_depot, Y_depot, 'ko', alpha=0.)
        return

    def draw_service_soc(self, network: Network.Network, ax: plt.axes):
        X_cs = [k for k, _ in enumerate(self.S) if network.is_charging_station(self.S[k])]
        draw_soc_arrows(self, X_cs, ax, 'Crimson', 'Recharging battery')
        return

    def draw_travelling_soc(self, ax: plt.Axes):
        preX = list(range(len(self.S)))
        X = preX[:-1]
        Y = self.eos_state[1, :-1]
        U = np.ones_like(preX[1:])
        V = self.sos_state[1, 1:] - Y

        ax.quiver(X, Y, U, V, scale=1, angles='xy', scale_units='xy', color='SteelBlue', width=0.0004 * 16, zorder=15,
                  label='Travelling')
        return

    def draw_soc_policy(self, ax: plt.Axes):
        ax.axhline(self.alpha_up, linestyle='--', color='black', label='SOH policy')
        ax.axhline(self.alpha_down, linestyle='--', color='black', label='SOH policy')
        ax.fill_between([-1, len(self.S)], self.alpha_down, self.alpha_up, color='lightgrey', alpha=.35)
        return

    def draw_travelling_payload(self, ax: plt.Axes):
        preX = list(range(len(self.S)))
        X = preX[:-1]
        Y = self.eos_state[2, :-1]
        U = np.ones_like(preX[1:])
        V = self.sos_state[2, 1:] - Y

        ax.quiver(X, Y, U, V, scale=1, angles='xy', scale_units='xy', color='SteelBlue', width=0.0004 * 16, zorder=15,
                  label='Travelling')
        return

    def draw_service_payload(self, ax: plt.Axes):
        X = list(range(len(self.S)))
        Y = self.sos_state[2, :]
        V = self.eos_state[2, :] - Y
        U = np.zeros_like(V)
        ax.quiver(X, Y, U, V, scale=1, angles='xy', scale_units='xy', color='SeaGreen', width=0.0004 * 16, headwidth=6,
                  zorder=10, label='Serving customer')

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
DETERMINISTIC EV WITH WAITING TIMES CLASS
"""


def pre_service_waiting_time(eos_time, tt, lower_tw):
    arrival_time = eos_time + tt
    pre_wt = lower_tw - arrival_time if lower_tw > arrival_time else 0.0
    return pre_wt


def post_service_waiting_time(eos_time, pre_wt, tt_with_pre_wt, lower_tw):
    """
    Calculates post service time
    @param eos_time: EOS time
    @param pre_wt: pre service waiting time
    @param tt_with_pre_wt: travel time considering that the EV departs at tod eos_time + pre_wt
    @param lower_tw: lower time window of target node
    @return: post-waiting time and residual pre-waiting time
    """
    post_wt = pre_wt
    residual_pre_wt = pre_service_waiting_time(eos_time + pre_wt, tt_with_pre_wt, lower_tw)
    return post_wt, residual_pre_wt


class ElectricVehicleWithWaitingTimes(ElectricVehicle):
    def set_route(self, S: Tuple[int, ...], L: Tuple[float, ...], x1_0: float, x2_0: float, x3_0: float,
                  first_pwt: float = 0.):
        super(ElectricVehicleWithWaitingTimes, self).set_route(S, L, x1_0, x2_0, x3_0)
        self.post_service_waiting_time[0] = first_pwt

    def step(self, network: Network.DeterministicCapacitatedNetwork, g=9.8):
        S, L = self.S, self.L
        Sk0, Sk1, Lk0, Lk1, k = S[0], S[1], L[0], L[1], 0
        for k, (Sk0, Sk1, Lk0, Lk1) in enumerate(zip(S[:-1], S[1:], L[:-1], L[1:])):
            # Calculate EOS time
            spent_time = network.spent_time(Sk0, self.sos_state[1, k], Lk0)
            post_wt = self.post_service_waiting_time[k]
            eos_time = self.sos_state[0, k] + spent_time + post_wt

            # Obtain pre and post waiting times
            edge = network.edges[Sk0][Sk1]
            m = self.weight + self.sos_state[2, k] - network.demand(Sk0)  # kg
            lower_tw = network.time_window_low(Sk1)

            t_ij = self.tt(eos_time, edge)
            E_ij = self.Ec(eos_time, m, g, edge)

            pre_wt = pre_service_waiting_time(eos_time, t_ij, lower_tw)

            if pre_wt:  # EV has to wait. Check post wt
                _t_ij = self.tt(eos_time + pre_wt, edge)
                _E_ij = self.Ec(eos_time + pre_wt, m, g, edge)
                to_add_post_wt, residual_pre_wt = post_service_waiting_time(eos_time, pre_wt, _t_ij, lower_tw)
                post_wt += to_add_post_wt

                if _E_ij < E_ij:  # Post wt provides lower energy consumption
                    pre_wt = residual_pre_wt
                    t_ij = _t_ij
                    E_ij = _E_ij

                else:
                    post_wt = 0.

            e_ij = 100 * E_ij / self.battery_capacity

            # Update EOS state
            f0k = np.array([spent_time, Lk0, -network.demand(Sk0)])
            self.eos_state[:, k] = self.sos_state[:, k] + f0k

            # Update SOS state at next stop
            f1k = np.array([post_wt + t_ij + pre_wt, -e_ij, 0])
            self.sos_state[:, k + 1] = self.eos_state[:, k] + f1k

            # Update containers
            self.travel_times[k] = t_ij
            self.energy_consumption[k] = e_ij
            self.charging_times[k] = spent_time if network.is_charging_station(Sk0) else 0.
            self.service_time[k] = spent_time
            self.post_service_waiting_time[k] = post_wt
            self.pre_service_waiting_time[k + 1] = pre_wt

        # Update what happens at the end
        k += 1
        spent_time = network.spent_time(Sk0, self.sos_state[1, k], Lk0)
        f0k = np.array([spent_time, Lk0, -network.demand(Sk0)])
        self.eos_state[:, k] = self.sos_state[:, k] + f0k

    def plot_operation(self, network: Network.Network, **kwargs):
        fig, (ax1, ax2, ax3) = super(ElectricVehicleWithWaitingTimes, self).plot_operation(network, **kwargs)
        self.draw_waiting_times(ax1)
        return fig, (ax1, ax2, ax3)

    def draw_waiting_times(self, ax: plt.Axes):
        X = list(range(len(self.S)))

        Y_pre = self.sos_state[0, :] - self.pre_service_waiting_time
        U = np.zeros_like(X)
        V = self.pre_service_waiting_time
        ax.quiver(X, Y_pre, U, V, scale=1, angles='xy', scale_units='xy', color='Orange', width=0.0004 * 16,
                  headwidth=6, zorder=10)

        Y_post = self.eos_state[0, :]
        U = np.zeros_like(X)
        V = self.post_service_waiting_time
        ax.quiver(X, Y_post, U, V, scale=1, angles='xy', scale_units='xy', color='Orange', width=0.0004 * 16,
                  headwidth=6, zorder=10)
        return


"""
GAUSSIAN ELECTRIC VEHICLE CLASS
"""


def probability_in_node(probability_container: np.ndarray, mu: float, sigma: float, spent_time: float,
                        do_evaluation_container: np.ndarray, dt=None):
    """
    Calculates the probability an EV is a node using Gaussian statistics
    @param probability_container: an array that will store the probabilities for each TOD
    @param mu: expected time the EV arrives at the node
    @param sigma: std of the time the EV arrives at the node
    @param spent_time: the service time at the node
    @param do_evaluation_container: array of ones and zeros. When 1, evaluate; whereas, when zero, do not evaluate
    @param dt: sample time. Default: None (calculates it using the length of the container array)
    @return: None. The probability container is modified in place to store the probability values
    """
    if dt is None:
        dt = int(24 * 60 * 60 / len(probability_container))
    t = 0
    for i, (_, evaluate) in enumerate(zip(probability_container, do_evaluation_container)):
        if evaluate:
            cdf1 = normal_cdf(t, mu, sigma)
            cdf2 = 1 - normal_cdf(t, mu + spent_time, sigma)
            probability_container[i] += cdf1 * cdf2
        t += dt
    return


@dataclass
class GaussianElectricVehicle(ElectricVehicle):
    state_reaching_covariance: Union[np.ndarray, None] = None
    state_leaving_covariance: Union[np.ndarray, None] = None
    Q: Union[np.ndarray, None] = None
    visits_a_cs: bool = False
    visited_nodes: Union[np.ndarray, None] = None
    probability_in_node_container: Union[np.ndarray, None] = None
    sample_time_probability_in_cs: float = 5.0
    num_customers: int = 0
    evaluate_all_container: Union[np.ndarray, None] = None

    def __post_init__(self):
        super(GaussianElectricVehicle, self).__post_init__()
        self.Q = np.zeros((3, 3))

    def reset(self, network: Network.GaussianCapacitatedNetwork):
        super(GaussianElectricVehicle, self).reset(network)
        self.state_reaching_covariance = None
        self.state_leaving_covariance = None
        self.reset_visited_nodes_array(network)

    def create_visited_nodes_array(self, network: Network.GaussianCapacitatedNetwork):
        self.visited_nodes = np.zeros(len(network))

    def setup(self, network_size: int, sample_time: float):
        self.sample_time_probability_in_cs = sample_time
        size = (network_size, int(24 * 60 * 60 / self.sample_time_probability_in_cs))
        self.probability_in_node_container = np.zeros(size)
        self.evaluate_all_container = np.ones(size)

    def reset_visited_nodes_array(self, network: Network.GaussianCapacitatedNetwork):
        if self.visited_nodes is None:
            self.create_visited_nodes_array(network)
        self.visited_nodes.fill(0)

    def reset_probability_in_cs_array(self):
        self.probability_in_node_container.fill(0)

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
            spent_time = network.spent_time(Sk0, self.sos_state[1, k], Lk0)
            f0k = np.array([spent_time, Lk0, -network.demand(Sk0)])
            self.eos_state[:, k] = self.sos_state[:, k] + f0k

            edge = network.edges[Sk0][Sk1]
            v_ij, sigma_ij = network.v(Sk0, Sk1, self.eos_state[0, k])  # m/s
            m = self.weight + self.eos_state[2, k]  # kg

            a1 = self.rho_air * self.Af * self.Cd * edge.length * v_ij ** 2 / 2
            a2 = edge.road_cos_length * g * self.Cr * self.c1 * m * v_ij / 1000
            a3 = edge.road_sin_length * g * m
            b1 = self.rho_air * self.Af * self.Cd * edge.length * v_ij
            b2 = edge.road_cos_length * g * self.Cr * self.c1 * m / 1000

            mu_E_ij = a1 + a2 + a3  # J
            sigma_E_ij = (b1 + b2) * sigma_ij  # J

            mu_e_ij = 100 * mu_E_ij / self.battery_capacity
            sigma_e_ij = 100 * sigma_E_ij / self.battery_capacity

            mu_t_ij = edge.length / v_ij if v_ij else 0.0  # s
            sigma_t_ij = -edge.length * sigma_ij / v_ij ** 2 if v_ij else 0.0  # s

            f1k = np.array([mu_t_ij, -mu_e_ij, 0])
            self.sos_state[:, k + 1] = self.eos_state[:, k] + f1k

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

            # self.visited_nodes[Sk0] += 1
            if network.is_charging_station(Sk0):
                self.visits_a_cs = True

        # Update what happens at the end (leaving)
        spent_time = network.spent_time(Sk1, self.sos_state[1, k], Lk1)
        f0k = np.array([spent_time, Lk1, -network.demand(Sk1)])
        self.service_time[k + 1] = spent_time
        self.eos_state[:, k + 1] = self.sos_state[:, k + 1] + f0k
        # self.visited_nodes[Sk1] += 1

    def probability_in_node(self, node_id: int, do_evaluation_container: np.ndarray):
        """
        Calculates the probability of being in a node
        @param node_id: the node id
        @param do_evaluation_container: array of ones and zeros that represent when to evaluate
        @return: None. Results are stored in self.probability_in_node[node_id, :]
        """
        if do_evaluation_container is None:
            do_evaluation_container = self.evaluate_all_container
        position_in_S = [k for k, Sk in enumerate(self.S) if Sk == node_id]
        # sample_time = self.sample_time_probability_in_cs
        for k in position_in_S:
            mu = self.sos_state[0, k]
            sigma = np.sqrt(self.state_reaching_covariance[0, 3 * k])  # TODO is this better than math.sqrt(.)?
            spent_time = self.service_time[k]
            probability_in_node(self.probability_in_node_container[node_id, :], mu, sigma, spent_time,
                                do_evaluation_container, self.sample_time_probability_in_cs)
        return

    def check_feasibility(self, network: Network.Network, penalization_constant=0.) -> float:
        self.penalization = 0.0
        self.constraint_max_tour_time()
        self.constraint_max_payload()
        self.constraint_time_window_and_soc_policy(network)
        if self.penalization > 0.:
            self.penalization += penalization_constant
        return self.penalization

    def constraint_max_tour_time(self, PRB=.9545):
        MU = self.sos_state[0, -1]
        SIG = np.sqrt(self.state_reaching_covariance[0, -3])
        VAL = self.max_tour_duration + self.x1_0
        CDF = normal_cdf(VAL, MU, SIG)
        self.penalization += penalization_stochastic(CDF, PRB, w=1e4)

    def constraint_max_payload(self):
        if self.eos_state[2, 0] > self.max_payload:
            self.penalization += penalization_deterministic(self.eos_state[2, 0], self.max_payload)

    def constraint_time_window_and_soc_policy(self, network: Network.Network):
        for k, (Sk, Lk) in enumerate(zip(self.S, self.L)):
            node = network.nodes[Sk]
            if node.is_customer():
                """
                TIME WINDOW - LOWER BOUND
                """
                PRB = .9545
                MU = self.sos_state[0, k]
                SIG = np.sqrt(self.state_reaching_covariance[0, 3 * k])
                VAL = node.time_window_low
                CDF = 1 - normal_cdf(VAL, MU, SIG)
                self.penalization += penalization_stochastic(CDF, PRB, w=1e4)

                """
                TIME WINDOW - UPPER BOUND
                """
                PRB = .9545
                MU = self.eos_state[0, k]
                SIG = np.sqrt(self.state_reaching_covariance[0, 3 * k])
                VAL = node.time_window_upp
                CDF = normal_cdf(VAL, MU, SIG)
                self.penalization += penalization_stochastic(CDF, PRB, w=1e4)

            """
            SOC BOUND - REACHING
            """
            PRB = .9545
            MU = self.sos_state[1, k]
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
            MU = self.eos_state[1, k]
            SIG = SIG
            VAL1 = self.alpha_up
            VAL2 = self.alpha_down
            CDF1 = normal_cdf(VAL1, MU, SIG)
            CDF2 = normal_cdf(VAL2, MU, SIG)
            CDF = CDF1 - CDF2
            self.penalization += penalization_stochastic(CDF, PRB, w=1e4)

    def draw_travelling_times(self, ax: plt.Axes):
        super(GaussianElectricVehicle, self).draw_travelling_times(ax)
        Q = self.state_reaching_covariance
        X = list(range(len(self.S)))
        Y_reaching_std = np.array([np.sqrt(q) for i, q in enumerate(Q[0, :]) if i % 3 == 0])
        r_time = self.sos_state[0, :]
        l_time = self.eos_state[0, :]
        ax.fill_between(X, r_time - 3 * Y_reaching_std, r_time + 3 * Y_reaching_std, color='red', alpha=0.2)

        Y_leaving_std = np.array([np.sqrt(q) for i, q in enumerate(Q[0, :]) if i % 3 == 0])
        ax.fill_between(X, l_time - 3 * Y_leaving_std, l_time + 3 * Y_leaving_std, color='green', alpha=0.2)

    def draw_travelling_soc(self, ax: plt.Axes):
        super(GaussianElectricVehicle, self).draw_travelling_soc(ax)
        Q = self.state_reaching_covariance
        X = list(range(len(self.S)))
        Y_reaching_std = np.array([np.sqrt(q) for i, q in enumerate(Q[1, 1:]) if i % 3 == 0])
        r_soc = self.sos_state[1, :]
        l_soc = self.eos_state[1, :]
        ax.fill_between(X, r_soc - 3 * Y_reaching_std, r_soc + 3 * Y_reaching_std, color='red', alpha=0.2)

        Y_leaving_std = np.array([np.sqrt(q) for i, q in enumerate(Q[1, 1:]) if i % 3 == 0])
        ax.fill_between(X, l_soc - 3 * Y_leaving_std, l_soc + 3 * Y_leaving_std, color='green', alpha=0.2)

    def xml_element(self, assign_customers=False, with_route=False, this_id=None):
        element = super(GaussianElectricVehicle, self).xml_element(assign_customers, with_route, this_id)
        if with_route:
            _route = element.find('route')  # TODO add covariance matrix
        return element

    @classmethod
    def from_xml_element(cls, element: ET.Element):
        instance = super(GaussianElectricVehicle, cls).from_xml_element(element)
        return instance


"""
USEFUL FUNCTIONS
"""


def travel_time(v: float, d: float):
    tt = d / v if v else 0.0
    return tt


def energy_consumption(v, d, m, Cr, c1, c2, g, Af, Cd, rho_air, road_cos_length, road_sin_length):
    F = road_cos_length * g * Cr * 1e-3 * (c1 * v + c2) + road_sin_length * g
    G = rho_air * Af * Cd * v ** 2 * d / 2.
    Ec = m * F + G  # J
    return Ec


def from_xml_element(element: ET.Element, ev_type: Union[object, str] = None):
    if ev_type is None:
        cls = globals()[element.get('type')]
    elif type(ev_type) is str:
        cls = globals()[ev_type]
    else:
        cls = ev_type
    return cls.from_xml_element(element)


def draw_service_time_arrows(ev: ElectricVehicle, X: list, ax: plt.Axes, color: str, label: str):
    Y = [ev.sos_state[0, k] for k in X]
    V = [ev.service_time[k] for k in X]
    U = np.zeros_like(V)
    ax.quiver(X, Y, U, V, scale=1, angles='xy', scale_units='xy', color=color, width=0.0004 * 16, headwidth=6,
              zorder=10, label=label)


def draw_soc_arrows(ev: ElectricVehicle, X: list, ax: plt.Axes, color: str, label: str):
    Y = [ev.sos_state[1, k] for k in X]
    V = [ev.L[k] for k in X]
    U = np.zeros_like(V)
    ax.quiver(X, Y, U, V, scale=1, angles='xy', scale_units='xy', color=color, width=0.0004 * 16, headwidth=6,
              zorder=10, label=label)


def place_labels(ev: ElectricVehicle, X: list, ax: plt.Axes, x_off=0.0, y_off=0.0):
    return


