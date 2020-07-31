import time

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import xml.dom.minidom
from copy import deepcopy

# Visualization tools
from bokeh.models import Whisker, Span, Range1d
from bokeh.models.annotations import Arrow, Label
from bokeh.models.arrow_heads import VeeHead
from bokeh.plotting import figure, show

import models.Network as net
from models.ElectricVehicle import *
from res.IOTools import write_pretty_xml


# CLASSES
class TupleIndex(list):
    def __add__(self, other: int):
        i = 2 * self[0] + self[1] + other
        return TupleIndex(integer_to_tuple2(i))


# FUNCTIONS
def distance(results, multi: np.ndarray, b: np.ndarray):
    return sum([dist_fun(multi[i], b[i]) for i, result in enumerate(results) if not result])


def dist_fun(x: Union[ndarray, float], y: Union[ndarray, float]):
    # return np.abs(x - y)
    # return np.sqrt(np.power(x - y, 2))
    # return np.abs(np.power(x - y, 2))
    return (x - y) ** 2


def integer_to_tuple2(k: int):
    val = int(k / 2)
    if k % 2:
        return val, 1
    return val, 0


def theta_vector_in_array(init_state, time_vectors, network_size, events_count, op_vector, theta_index) -> None:
    iTheta0, iTheta1 = theta_index, theta_index + network_size
    op_vector[iTheta0:iTheta1] = init_state
    counters = [[TupleIndex([0, 0]), 0] for _ in range(len(time_vectors))]

    for k in range(1, events_count):
        iTheta0 += network_size
        iTheta1 += network_size
        min_t, ind_ev = min([(time_vectors[ind][x[0][1]][x[0][0]], ind) for ind, x in enumerate(counters) if not x[1]])
        event = 1 if counters[ind_ev][0][1] else -1
        node = time_vectors[ind_ev][2][counters[ind_ev][0][0] + counters[ind_ev][0][1]]
        op_vector[iTheta0:iTheta1] = op_vector[iTheta0 - network_size:iTheta1 - network_size]
        op_vector[iTheta0 + node] += event
        counters[ind_ev][0] = counters[ind_ev][0] + 1
        if not counters[ind_ev][0][1] and len(time_vectors[ind_ev][2]) - 1 == counters[ind_ev][0][0] + \
                counters[ind_ev][0][1]:
            counters[ind_ev][1] = 1


def theta_matrix(matrix, time_vectors, events_count) -> None:
    counters = [[TupleIndex([0, 0]), 0] for _ in range(len(time_vectors))]

    for k in range(1, events_count):
        min_t, ind_ev = min([(time_vectors[ind][x[0][1]][x[0][0]], ind) for ind, x in enumerate(counters) if not x[1]])
        event = 1 if counters[ind_ev][0][1] else -1
        node = time_vectors[ind_ev][2][counters[ind_ev][0][0] + counters[ind_ev][0][1]]
        matrix[:, k] = matrix[:, k - 1]
        matrix[node, k] += event
        counters[ind_ev][0] = counters[ind_ev][0] + 1
        if not counters[ind_ev][0][1] and len(time_vectors[ind_ev][2]) - 1 == counters[ind_ev][0][0] + \
                counters[ind_ev][0][1]:
            counters[ind_ev][1] = 1


class Fleet:
    vehicles: Dict[int, ElectricVehicle]
    network: net.Network
    vehicles_to_route: Tuple[int, ...]
    theta_vector: Union[np.ndarray, None]
    optimization_vector: Union[np.ndarray, None]
    optimization_vector_indices: Union[Tuple, None]
    starting_points: Dict[int, InitialCondition]
    route: bool

    def __init__(self, vehicles=None, network=None, vehicles_to_route=None, route=True):
        self.set_vehicles(vehicles)
        self.set_network(network)
        self.set_vehicles_to_route(vehicles_to_route)

        self.theta_vector = None
        self.theta_matrix = None
        self.optimization_vector = None
        self.optimization_vector_indices = None

        self.starting_points = {}
        self.route = route

    def __len__(self):
        return len(self.vehicles)

    def set_vehicles(self, vehicles: Dict[int, ElectricVehicle]) -> None:
        self.vehicles = vehicles

    def set_network(self, network: Network) -> None:
        self.network = network

    def drop_vehicle(self, ev_id: int) -> None:
        del self.vehicles[ev_id]
        self.vehicles_to_route = tuple(i for i in self.vehicles.keys())

    def resize_fleet(self, new_size, as_new=True, based_on=0, auto=False):
        if auto:
            net = self.network
            new_size = int(sum([net.demand(i) for i in net.customers]) / self.vehicles[0].max_payload) + 1

        ev_base = deepcopy(self.vehicles[based_on])
        if as_new:
            ev_base.reset()

        self.vehicles = {}
        for i in range(new_size):
            ev = deepcopy(ev_base)
            ev.id = i
            self.vehicles[i] = ev

        self.vehicles_to_route = tuple(i for i in range(new_size))

    def new_soc_policy(self, alpha_down, alpha_upp):
        for ev in self.vehicles.values():
            ev.alpha_down = alpha_down
            ev.alpha_up = alpha_upp

    def relax_time_windows(self):
        for cust in self.network.customers:
            self.network.nodes[cust].time_window_low = 0.
            self.network.nodes[cust].time_window_upp = 60 * 24

    def modify_cs_capacities(self, new_capacity):
        for cs in self.network.charging_stations:
            self.network.nodes[cs].capacity = new_capacity

    def assign_customers_in_route(self):
        for ev in self.vehicles.values():
            ev.assign_customers_in_route(self.network)

    def set_vehicles_to_route(self, vehicles: List[int]) -> None:
        if vehicles:
            self.vehicles_to_route = tuple(vehicles)

    def iterate(self):
        for ev in self.vehicles.values():
            route = ev.route
            dep_time, dep_soc, dep_pay = ev.x1_0, ev.x2_0, ev.x3_0
            ev.set_route(route, dep_time, dep_soc, dep_pay)
            ev.step(self.network)
        self.iterate_cs_capacities(None)

    def set_routes_of_vehicles(self, routes: RouteDict, iterate=True, iterate_cs=True,
                               init_theta: ndarray = None) -> None:
        for id_ev, (route, dep_time, dep_soc, dep_pay) in routes.items():
            self.vehicles[id_ev].set_route(route, dep_time, dep_soc, dep_pay)
            if iterate:
                self.vehicles[id_ev].step(self.network)
        if iterate_cs:
            self.iterate_cs_capacities(init_theta)

    def iterate_cs_capacities(self, init_theta: ndarray = None):
        sum_si = sum([len(ev.route[0]) for ev in self.vehicles.values()])
        num_cs = len(self.network.charging_stations)
        m = len(self.vehicles)
        num_events = 2 * sum_si - 2 * m + 1
        time_vectors = []
        for id_ev, ev in self.vehicles.items():
            time_vectors.append((self.vehicles[id_ev].state_leaving[0, :-1] - ev.waiting_times0[:-1],
                                 self.vehicles[id_ev].state_reaching[0, 1:], self.vehicles[id_ev].route[0]))
        if init_theta is None:
            init_theta = np.zeros(len(self.network))
            init_theta[0] = len(self)

        self.theta_matrix = np.zeros((len(self.network), num_events))
        self.theta_matrix[:, 0] = init_theta
        theta_matrix(self.theta_matrix, time_vectors, num_events)

    def cost_function(self) -> Tuple:
        cost_tt = sum([sum(ev.travel_times) for ev in self.vehicles.values()])
        cost_ec = sum([sum(ev.energy_consumption) for ev in self.vehicles.values()])
        cost_chg_op = sum([sum(ev.charging_times) for ev in self.vehicles.values()])
        cost_chg_cost = sum([sum(ev.route[1]) for ev in self.vehicles.values()])
        cost_wait_time = sum([sum(ev.waiting_times) for ev in self.vehicles.values()])
        return cost_tt, cost_ec, cost_chg_op, cost_chg_cost, cost_wait_time

    def feasible(self, online=False) -> Tuple[bool, Union[int, float], bool]:
        # 1. Variables to return
        is_feasible = True
        accept = True
        dist = 0

        # 2. Variables from the optimization vector and vehicles
        network = self.network
        m = len(self.vehicles)
        sum_si = sum([len(ev.route[0]) for ev in self.vehicles.values()])
        num_events = 2 * sum_si - 2 * m + 1
        num_cust = len(network.customers)

        for ev in self.vehicles.values():
            if ev.state_reaching[0, -1] - ev.state_leaving[0, 0] > ev.max_tour_duration:
                d = dist_fun(ev.max_tour_duration, ev.state_reaching[0, -1] - ev.state_leaving[0, 0])
                dist += d
                accept = False if d > 25 else accept

            if ev.state_leaving[2, 0] > ev.max_payload and not online:
                dist += dist_fun(ev.state_leaving[2, 0], ev.max_payload)

            for k, (Sk, Lk) in enumerate(zip(ev.route[0], ev.route[1])):
                if network.isCustomer(Sk):
                    node = network.nodes[Sk]
                    if node.time_window_low > ev.state_reaching[0, k]:
                        d = dist_fun(ev.state_reaching[0, k], node.time_window_low)
                        dist += 20*d
                        accept = False if d > 20 else accept

                    if node.time_window_upp < ev.state_leaving[0, k] - ev.waiting_times0[k]:
                        d = dist_fun(node.time_window_upp, ev.state_leaving[0, k] - ev.waiting_times0[k])
                        dist += 20*d
                        accept = False if d > 20 else accept

                if ev.state_reaching[1, k] < ev.alpha_down:
                    dist += dist_fun(ev.state_reaching[1, k], ev.alpha_down)

                if ev.state_reaching[1, k] > ev.alpha_up:
                    dist += dist_fun(ev.state_reaching[1, k], ev.alpha_up)

                if ev.state_leaving[1, k] < ev.alpha_down:
                    dist += dist_fun(ev.state_leaving[1, k], ev.alpha_down)

                if ev.state_leaving[1, k] > ev.alpha_up:
                    dist += dist_fun(ev.state_leaving[1, k], ev.alpha_up)

                if ev.state_reaching[1, k] < 0:
                    dist += dist_fun(ev.state_reaching[1, k], ev.alpha_down) + 1000
                    accept = False

                if ev.state_reaching[1, k] > 100:
                    dist += dist_fun(ev.state_reaching[1, k], ev.alpha_up) + 1000
                    accept = False

                if ev.state_leaving[1, k] < 0:
                    dist += dist_fun(ev.state_leaving[1, k], ev.alpha_down) + 1000
                    accept = False

                if ev.state_leaving[1, k] > 100:
                    dist += dist_fun(ev.state_leaving[1, k], ev.alpha_up) + 1000
                    accept = False

        for i, cs in enumerate(network.charging_stations):
            for k in range(num_events):
                if self.theta_matrix[1 + num_cust + i, k] > network.nodes[cs].capacity:
                    dist += dist_fun(self.theta_matrix[1 + num_cust + i, k], network.nodes[cs].capacity)
                    accept = False

        if dist > 0:
            is_feasible = False

        return is_feasible, dist, accept

    def save_operation_xml(self, path: str, critical_points: Dict[int, Tuple[int, float, float, float]], pretty=False):
        # Open XML file
        tree = ET.parse(path)
        _info: ET = tree.find('info')
        _network: ET = tree.find('network')
        _fleet: ET = tree.find('fleet')
        _technologies: ET = _network.find('technologies')

        # Fleet data
        for _vehicle in _fleet:
            # Just do vehicles with valid critical points
            ev_id = int(_vehicle.get('id'))
            if ev_id in self.vehicles_to_route:
                # Remove all previous routes
                for _prev_route in _vehicle.findall('previous_route'):
                    _vehicle.remove(_prev_route)

                # Remove all previous critical points
                for _crit_point in _vehicle.findall('critical_point'):
                    _vehicle.remove(_crit_point)

                # Save new route
                _previous_route = ET.SubElement(_vehicle, 'previous_route')
                for Sk, Lk in zip(self.vehicles[ev_id].route[0], self.vehicles[ev_id].route[1]):
                    attrib = {'Sk': str(Sk), 'Lk': str(Lk)}
                    _node = ET.SubElement(_previous_route, 'node', attrib=attrib)

                critical_point = critical_points[ev_id]
                attrib_cp = {'k': str(critical_point[0]), 'x1': str(critical_point[1]), 'x2': str(critical_point[2]),
                             'x3': str(critical_point[3])}
                _critical_point = ET.SubElement(_vehicle, 'critical_point', attrib=attrib_cp)

        tree.write(path)
        if pretty:
            write_pretty_xml(path)
        return tree

    def update_from_xml(self, path, do_network=False) -> ET:
        # Open XML file
        tree = ET.parse(path)
        _info: ET = tree.find('info')
        _network: ET = tree.find('network')
        _fleet: ET = tree.find('fleet')
        _technologies: ET = _network.find('technologies')

        # Network data
        if do_network:
            self.set_network(net.from_element_tree(tree))

        # Fleet data
        self.starting_points = {}
        vehicles_to_route = []
        for _vehicle in _fleet:
            ev_id = int(_vehicle.get('id'))
            _critical_point = _vehicle.find('critical_point')
            k = int(_critical_point.get('k'))
            if k != -1:
                vehicles_to_route.append(ev_id)
                _previous_sequence = _vehicle.find('previous_route')
                previous_sequence = [[int(x.get('Sk')) for x in _previous_sequence],
                                     [float(x.get('Lk')) for x in _previous_sequence]]

                new_customers = tuple(x for x in previous_sequence[0][k + 1:] if self.network.isCustomer(x))
                self.vehicles[ev_id].set_customers_to_visit(new_customers)

                S0 = previous_sequence[0][k]
                L0 = previous_sequence[1][k]
                x1_0 = float(_critical_point.get('x1'))
                x2_0 = float(_critical_point.get('x2'))
                x3_0 = float(_critical_point.get('x3'))
                self.starting_points[ev_id] = InitialCondition(S0, L0, x1_0, x2_0, x3_0)

                route_from_crit_point = (tuple(previous_sequence[0][k:]), tuple(previous_sequence[1][k:]))
                self.vehicles[ev_id].set_route(route_from_crit_point, x1_0, x2_0, x3_0)

        self.set_vehicles_to_route(vehicles_to_route)

        return tree

    def plot_operation_pyplot(self, arrow_colors=('SteelBlue', 'Crimson', 'SeaGreen'), fig_size=(16, 5),
                              label_offset=(.15, -6), subplots=True, save_to=None):
        figs = []
        for id_ev, vehicle in self.vehicles.items():
            Si, Li = vehicle.route[0], vehicle.route[1]
            wt0, wt1 = vehicle.waiting_times0, vehicle.waiting_times1
            st_reaching, st_leaving = vehicle.state_reaching, vehicle.state_leaving
            r_time, r_soc, r_payload = st_reaching[0, :], st_reaching[1, :], st_reaching[2, :]
            l_time, l_soc, l_payload = st_leaving[0, :], st_leaving[1, :], st_leaving[2, :]

            si = len(Si)

            fig = plt.figure(figsize=fig_size)

            ### FIG X1 ###
            plt.subplot(131)
            plt.grid()
            # time windows
            X_tw = [k for k, Sk in enumerate(Si) if self.network.isCustomer(Sk)]
            Y_tw = [(self.network.nodes[Sk].time_window_upp + self.network.nodes[Sk].time_window_low) / 2.0
                    for k, Sk in enumerate(Si) if self.network.isCustomer(Sk)]
            tw_sigma = [(self.network.nodes[Sk].time_window_upp - self.network.nodes[Sk].time_window_low) / 2.0
                        for k, Sk in enumerate(Si) if self.network.isCustomer(Sk)]
            plt.errorbar(X_tw, Y_tw, tw_sigma, ecolor='black', fmt='none', capsize=6, elinewidth=1, zorder=5,
                         label='Customer time window')

            # time in nodes customers
            X_node = [i for i in range(si) if self.network.isCustomer(Si[i])]
            Y_node = [t for i, t in enumerate(r_time) if self.network.isCustomer(Si[i])]
            U_node = [0 for i in range(si) if self.network.isCustomer(Si[i])]
            V_node = [l_time[i] - r_time[i] - wt0[i] for i in range(si) if self.network.isCustomer(Si[i])]
            plt.quiver(X_node, Y_node, U_node, V_node, scale=1, angles='xy', scale_units='xy',
                       color=arrow_colors[2], width=0.0004 * fig_size[0], headwidth=6, zorder=10,
                       label='Serving customer')

            # time in nodes CS
            X_node = [i for i in range(si) if self.network.isChargingStation(Si[i])]
            Y_node = [t for i, t in enumerate(r_time + wt1) if self.network.isChargingStation(Si[i])]
            U_node = [0 for i in range(si) if self.network.isChargingStation(Si[i])]
            V_node = [l_time[i] - r_time[i] - wt0[i] for i in range(si) if self.network.isChargingStation(Si[i])]
            plt.quiver(X_node, Y_node, U_node, V_node, scale=1, angles='xy', scale_units='xy',
                       color=arrow_colors[1], width=0.0004 * fig_size[0], headwidth=6, zorder=10,
                       label='Charging operation')

            # waiting times 0
            X_node = range(si)
            Y_node = l_time - wt0
            U_node = [0] * si
            V_node = wt0
            plt.quiver(X_node, Y_node, U_node, V_node, scale=1, angles='xy', scale_units='xy',
                       color='orange', width=0.0004 * fig_size[0], headwidth=6, zorder=10, label='Waiting')

            # waiting times 1
            X_node = range(si)
            Y_node = r_time - wt1
            U_node = [0] * si
            V_node = wt1
            plt.quiver(X_node, Y_node, U_node, V_node, scale=1, angles='xy', scale_units='xy',
                       color='orange', width=0.0004 * fig_size[0], headwidth=6, zorder=10)

            # travel time
            X_edge, Y_edge = range(si - 1), l_time[0:-1]
            U_edge, V_edge = np.ones(si - 1), r_time[1:] - l_time[0:-1] - wt1[1:]
            color_edge = [arrow_colors[0]] * si
            plt.quiver(X_edge, Y_edge, U_edge, V_edge, scale=1, angles='xy', scale_units='xy', color=color_edge,
                       width=0.0004 * fig_size[0], zorder=15, label='Travelling')

            # Maximum tour time
            plt.axhline(vehicle.state_leaving[0, 0] + vehicle.max_tour_duration, linestyle='--', color='black',
                        label='Maximum tour time')

            # Annotate nodes
            labels = [str(i) for i in vehicle.route[0]]
            pos = [(x, y) for x, y in zip(range(si), r_time)]
            for label, (x, y) in zip(labels, pos):
                plt.annotate(label, (x + label_offset[0], y + label_offset[1]))

            plt.legend(fontsize='small')
            plt.xlabel('Stop')
            plt.ylabel('Time of the day (min)')
            plt.title(f'Arrival and departure times (EV {id_ev})')
            plt.xlim(-.5, si - .5)

            ### FIG X2 ###
            plt.subplot(132)
            plt.grid()

            # SOC in Customer
            X_node = [i for i in range(si) if self.network.isCustomer(Si[i])]
            Y_node = [t for i, t in enumerate(r_soc) if self.network.isCustomer(Si[i])]
            U_node = [0 for i in range(si) if self.network.isCustomer(Si[i])]
            V_node = [l_soc[i] - r_soc[i] for i in range(si) if self.network.isCustomer(Si[i])]
            plt.quiver(X_node, Y_node, U_node, V_node, scale=1, angles='xy', scale_units='xy',
                       color=arrow_colors[2], width=0.0004 * fig_size[0], headwidth=6, zorder=10,
                       label='Serving customer')

            # SOC in CS
            X_node = [i for i in range(si) if self.network.isChargingStation(Si[i])]
            Y_node = [t for i, t in enumerate(r_soc) if self.network.isChargingStation(Si[i])]
            U_node = [0 for i in range(si) if self.network.isChargingStation(Si[i])]
            V_node = [l_soc[i] - r_soc[i] for i in range(si) if self.network.isChargingStation(Si[i])]
            plt.quiver(X_node, Y_node, U_node, V_node, scale=1, angles='xy', scale_units='xy',
                       color=arrow_colors[1], width=0.0004 * fig_size[0], headwidth=6, zorder=10,
                       label='Charging operation')

            # travel SOC
            X_edge, Y_edge = range(si - 1), l_soc[0:-1]
            U_edge, V_edge = np.ones(si - 1), r_soc[1:] - l_soc[0:-1]
            color_edge = [arrow_colors[0]] * si
            plt.quiver(X_edge, Y_edge, U_edge, V_edge, scale=1, angles='xy', scale_units='xy', color=color_edge,
                       width=0.0004 * fig_size[0], zorder=15, label='Travelling')

            # Annotate nodes
            labels = [str(i) for i in vehicle.route[0]]
            pos = [(x, y) for x, y in zip(range(si), r_soc)]
            for label, (x, y) in zip(labels, pos):
                plt.annotate(label, (x + label_offset[0], y + label_offset[1]))

            # SOH policy
            plt.axhline(vehicle.alpha_down, linestyle='--', color='black', label='SOH policy')
            plt.axhline(vehicle.alpha_up, linestyle='--', color='black', label=None)
            plt.fill_between([-1, si], vehicle.alpha_down, vehicle.alpha_up, color='lightgrey', alpha=.35)

            plt.legend(fontsize='small')

            # Scale yaxis from 0 to 100
            plt.ylim((0, 100))
            plt.xlim(-.5, si - .5)
            plt.xlabel('Stop')
            plt.ylabel('State Of Charge (%)')
            plt.title(f'EV Battery SOC (EV {id_ev})')

            ### FIG X3 ###
            plt.subplot(133)
            plt.grid()
            # payload in nodes
            X_node, Y_node = range(si), r_payload
            U_node, V_node = np.zeros(si), l_payload - r_payload
            color_node = [arrow_colors[2] if self.network.isCustomer(i) or self.network.isDepot(i)
                          else arrow_colors[1] for i in Si]
            plt.quiver(X_node, Y_node, U_node, V_node, scale=1, angles='xy', scale_units='xy', color=color_node,
                       width=0.0004 * fig_size[0], headwidth=6, zorder=10, label='Serving customer')

            # traveling payload
            X_edge, Y_edge = range(si - 1), l_payload[0:-1]
            U_edge, V_edge = np.ones(si - 1), r_payload[1:] - l_payload[0:-1]
            color_edge = [arrow_colors[0]] * si
            plt.quiver(X_edge, Y_edge, U_edge, V_edge, scale=1, angles='xy', scale_units='xy', color=color_edge,
                       width=0.0004 * fig_size[0], zorder=15, label='Travelling')

            # Annotate nodes FIXME not showing??
            labels = [str(i) for i in vehicle.route[0]]
            pos = [(x, y) for x, y in zip(range(si), r_payload)]
            for label, (x, y) in zip(labels, pos):
                plt.annotate(label, (x + label_offset[0], y + label_offset[1]))

            # Max weight
            plt.axhline(vehicle.max_payload, linestyle='--', color='black', label='Max. payload')

            plt.xlabel('Stop')
            plt.ylabel('Payload (t)')
            plt.title(f' (Payload evolution (EV {id_ev})')
            plt.legend(fontsize='small')

            if save_to is not None:
                fig.savefig(f'{save_to}operation_EV{id_ev}_full.pdf', format='pdf')

            figs.append(fig)

        # Charging stations occupation
        sum_si = sum([len(ev.route[0]) for ev in self.vehicles.values()])
        num_cs = len(self.network.charging_stations)
        m = len(self.vehicles)
        num_events = 2 * sum_si - 2 * m + 1
        events = range(num_events)

        fig = plt.figure(figsize=fig_size)
        plt.step(events, self.theta_matrix[-num_cs:, :].T)
        plt.title('CS Occupation')
        plt.xlabel('Event')
        plt.ylabel('Number of EVs')
        plt.legend(tuple(f'CS {i}' for i in self.network.charging_stations))
        figs.append(fig)
        return figs

    def draw_operation(self, color_route='red', **kwargs):
        fig, g = self.network.draw(**kwargs)
        pos = {i: (node.pos_x, node.pos_y) for i, node in self.network.nodes.items()}
        cc = 0
        for id_ev, ev in self.vehicles.items():
            edges = [(Sk0, Sk1) for Sk0, Sk1 in zip(ev.route[0][0:-1], ev.route[0][1:])]
            if type(color_route) == str:
                nx.draw_networkx_edges(g, pos, edgelist=edges, ax=fig.get_axes()[0], edge_color=color_route)
            else:
                nx.draw_networkx_edges(g, pos, edgelist=edges, ax=fig.get_axes()[0], edge_color=color_route[cc])
                cc = cc + 1 if cc + 1 < len(color_route) else 0
        return fig, g

    def plot_operation(self, save=False, path=None):
        # Vectors to plot
        colorArrowTravel = 'SteelBlue'
        colorArrowCharging = 'Crimson'
        colorArrowServing = 'SeaGreen'

        # Plot
        maxTw = -1
        minTw = 100000000000

        for id_ev, vehicle in self.vehicles.items():
            # Variables to use
            node_seq = vehicle.route[0]
            kVehicle = range(0, len(node_seq))
            state_reaching = vehicle.state_reaching
            state_leaving = vehicle.state_leaving

            # figures
            figX1 = figure(plot_width=480, plot_height=360,
                           title=f'Vehicle {id_ev}: arrival and departure times',
                           toolbar_location='right')
            figX2 = figure(plot_width=480, plot_height=360,
                           title=f'Vehicle {id_ev}: SOC evolution',
                           y_range=(0, 100),
                           toolbar_location='right')
            figX3 = figure(plot_width=480, plot_height=360,
                           title=f'Vehicle {id_ev}: weight evolution',
                           toolbar_location='right')

            # Iterate each node
            kCustomers = []
            tWindowsUpper = []
            tWindowsLower = []
            reaching_vector_prev = state_reaching[:, 0]
            leaving_vector_prev = state_leaving[:, 0]
            kPrev = 0
            for k, node in enumerate(node_seq):
                # Current vectors
                reaching_vector = state_reaching[:, k]
                leaving_vector = state_leaving[:, k]

                # Add node number
                label = Label(x=k, y=reaching_vector[0], y_offset=-5, text=str(node), text_baseline='top')
                figX1.add_layout(label)

                label = Label(x=k, y=reaching_vector[1], y_offset=10, text=str(node), text_baseline='top')
                figX2.add_layout(label)

                label = Label(x=k, y=reaching_vector[2], y_offset=10, text=str(node), text_baseline='top')
                figX3.add_layout(label)

                # Arrow color
                if self.network.isChargingStation(node):
                    colorArrowSpent = colorArrowCharging
                else:
                    colorArrowSpent = colorArrowServing

                # Time spent
                arrowSpent = Arrow(x_start=k, y_start=reaching_vector[0],
                                   x_end=k, y_end=leaving_vector[0],
                                   end=VeeHead(size=8, fill_color=colorArrowSpent, line_color=colorArrowSpent),
                                   line_color=colorArrowSpent, line_alpha=1)
                figX1.add_layout(arrowSpent)

                # Draw SOC increase arrows
                arrowSpent = Arrow(x_start=k, y_start=reaching_vector[1],
                                   x_end=k, y_end=leaving_vector[1],
                                   end=VeeHead(size=8, fill_color=colorArrowSpent, line_color=colorArrowSpent),
                                   line_color=colorArrowSpent, line_alpha=1, line_width=1.5, visible=True)
                figX2.add_layout(arrowSpent)

                # Delivery
                arrowSpent = Arrow(x_start=k, y_start=reaching_vector[2],
                                   x_end=k, y_end=leaving_vector[2],
                                   end=VeeHead(size=8, fill_color=colorArrowSpent, line_color=colorArrowSpent),
                                   line_color=colorArrowSpent, line_alpha=1, line_width=1.5, visible=True)
                figX3.add_layout(arrowSpent)

                # Draw travel arrows
                if k != 0:
                    # X1
                    arrowTravel = Arrow(x_start=kPrev, y_start=leaving_vector_prev[0],
                                        x_end=k, y_end=reaching_vector[0],
                                        end=VeeHead(size=8, fill_color=colorArrowTravel, line_color=colorArrowTravel),
                                        line_color=colorArrowTravel, line_alpha=1)
                    figX1.add_layout(arrowTravel)

                    # X2
                    arrowTravel = Arrow(x_start=kPrev, y_start=leaving_vector_prev[1],
                                        x_end=k, y_end=reaching_vector[1],
                                        end=VeeHead(size=8, fill_color=colorArrowTravel, line_color=colorArrowTravel),
                                        line_color=colorArrowTravel, line_alpha=1, line_width=1.5, visible=True)
                    figX2.add_layout(arrowTravel)

                    # X3

                    arrowTravel = Arrow(x_start=kPrev, y_start=leaving_vector_prev[2],
                                        x_end=k, y_end=reaching_vector[2],
                                        end=VeeHead(size=8, fill_color=colorArrowTravel, line_color=colorArrowTravel),
                                        line_color=colorArrowTravel, line_alpha=1, line_width=1.5, visible=True)
                    figX3.add_layout(arrowTravel)

                # Custumer time windows
                if self.network.isCustomer(node):
                    kCustomers.append(k)
                    node_instance = self.network.nodes[node]
                    tWindowsCenter = (node_instance.time_window_upp + node_instance.time_window_low) / 2.0
                    tWindowsWidth = (node_instance.time_window_upp - node_instance.time_window_low) / 2.0
                    tWindowsUpper.append(tWindowsCenter + tWindowsWidth)
                    tWindowsLower.append(tWindowsCenter - tWindowsWidth)
                    # Time windows whiskers
                    whiskerTW = Whisker(base=k, upper=tWindowsCenter + tWindowsWidth,
                                        lower=tWindowsCenter - tWindowsWidth)
                    figX1.add_layout(whiskerTW)
                    # update TW bounds
                    if tWindowsCenter + tWindowsWidth > maxTw:
                        maxTw = tWindowsCenter + tWindowsWidth
                    if tWindowsCenter - tWindowsWidth < minTw:
                        minTw = tWindowsCenter - tWindowsWidth

                # common
                reaching_vector_prev, leaving_vector_prev = reaching_vector, leaving_vector
                kPrev = k

            # horizontal line SOC
            hline1 = Span(location=vehicle.alpha_down, dimension='width', line_color='black')
            hline2 = Span(location=vehicle.alpha_up, dimension='width', line_color='black')
            figX2.renderers.extend([hline1, hline2])

            # adjust fig 1 to fit TWs
            figX1.y_range = Range1d(minTw - 10, maxTw + 10)

            figX1.line(kVehicle, vehicle.state_reaching[0, :], alpha=0)
            figX1.line(kVehicle, vehicle.state_leaving[0, :], alpha=0)
            figX2.line(kVehicle, vehicle.state_reaching[1, :], alpha=0)
            figX2.line(kVehicle, vehicle.state_leaving[1, :], alpha=0)
            figX3.line(kVehicle, vehicle.state_reaching[2, :], alpha=0)

            # Axes
            figX1.xaxis.axis_label = 'Vehicle stop k'
            figX1.yaxis.axis_label = 'Time of the day (min)'
            figX1.axis.axis_label_text_font_size = '15pt'
            figX1.axis.major_label_text_font_size = '13pt'
            figX1.title.text_font_size = '15pt'
            figX1.axis.axis_label_text_font = 'times'
            figX1.axis.major_label_text_font = 'times'
            figX1.title.text_font = 'times'
            figX1.title.align = 'center'

            figX2.xaxis.axis_label = 'Vehicle stop k'
            figX2.yaxis.axis_label = 'SOC (%)'
            figX2.axis.axis_label_text_font_size = '15pt'
            figX2.axis.major_label_text_font_size = '13pt'
            figX2.title.text_font_size = '15pt'
            figX2.axis.axis_label_text_font = 'times'
            figX2.axis.major_label_text_font = 'times'
            figX2.title.text_font = 'times'
            figX2.title.align = 'center'

            figX3.xaxis.axis_label = 'Vehicle stop k'
            figX3.yaxis.axis_label = 'Payload (ton)'
            figX3.axis.axis_label_text_font_size = '15pt'
            figX3.axis.major_label_text_font_size = '13pt'
            figX3.title.text_font_size = '15pt'
            figX3.axis.axis_label_text_font = 'times'
            figX3.axis.major_label_text_font = 'times'
            figX3.title.text_font = 'times'
            figX3.title.align = 'center'

            # Show plots of vehicles
            show(figX1)
            time.sleep(0.5)
            show(figX2)
            time.sleep(0.5)
            show(figX3)
            time.sleep(0.5)

        # Now, charging stations
        network_size = len(self.network)
        full_operation_theta_vector = self.optimization_vector[self.optimization_vector_indices[10]:]
        triggers = int(len(full_operation_theta_vector) / network_size)
        occupations = {x: [] for x in self.network.charging_stations}
        for i in range(triggers):
            theta_vector = full_operation_theta_vector[i * network_size:(i + 1) * network_size]
            for id_cs in self.network.charging_stations:
                occupations[id_cs].append(int(theta_vector[id_cs]))

        for id_cs in self.network.charging_stations:
            fig = figure(plot_width=600, plot_height=450,
                         title=f'Occupation of CS {id_cs}',
                         toolbar_location='right')
            fig.step(range(triggers), occupations[id_cs])

            fig.xaxis.axis_label = 'Event #'
            fig.yaxis.axis_label = 'EVs in charge stations'
            fig.axis.axis_label_text_font = 'times'
            fig.axis.axis_label_text_font_size = '15pt'
            fig.axis.major_label_text_font_size = '13pt'
            fig.title.text_font_size = '15pt'
            show(fig)
            time.sleep(0.5)
        return

    def xml_tree(self, assign_customers=False, with_routes=False, online=False):
        _fleet = ET.Element('fleet')
        for vehicle in self.vehicles.values():
            _fleet.append(vehicle.xml_element(assign_customers, with_routes, None, online))
        return _fleet

    def write_xml(self, path, network_in_file=False, assign_customers=False, with_routes=False, online=False,
                  print_pretty=False):
        tree = self.xml_tree(assign_customers, with_routes, online)
        if network_in_file:
            instance_tree = ET.Element('instance')
            instance_tree.append(self.network.xml_tree())
            instance_tree.append(tree)
            tree = instance_tree

        if print_pretty:
            xml_pretty = xml.dom.minidom.parseString(ET.tostring(tree, 'utf-8')).toprettyxml()
            with open(path, 'w') as file:
                file.write(xml_pretty)
        else:
            ET.ElementTree(tree).write(path)


def from_xml(path, assign_customers=False, with_routes=True, instance=True, from_online=False):
    # Open XML file
    tree = ET.parse(path)
    if instance:
        _fleet = tree.find('fleet')
    else:
        _fleet = tree.getroot()

    # Fleet data
    vehicles = {}
    vehicles_to_route = []
    for _vehicle in _fleet:
        ev_id = int(_vehicle.get('id'))
        max_tour_duration = float(_vehicle.get('max_tour_duration'))
        alpha_up = float(_vehicle.get('alpha_up'))
        alpha_down = float(_vehicle.get('alpha_down'))
        battery_capacity = float(_vehicle.get('battery_capacity'))
        battery_capacity_nominal = float(_vehicle.get('battery_capacity_nominal'))
        max_payload = float(_vehicle.get('max_payload'))
        weight = float(_vehicle.get('weight'))
        vehicles[ev_id] = ElectricVehicle(ev_id, weight, battery_capacity, battery_capacity_nominal, alpha_up,
                                          alpha_down, max_tour_duration, max_payload)
        if assign_customers:
            _assigned_customers = _vehicle.find('assigned_customers')
            assigned_customers = tuple(int(x.get('id')) for x in _assigned_customers)
            vehicles[ev_id].set_customers_to_visit(assigned_customers)

        if with_routes:
            _prev_route = _vehicle.find('online_route') if from_online else _vehicle.find('previous_route')
            try:
                x10, x20, x30 = float(_prev_route.get('x1')), float(_prev_route.get('x2')), float(_prev_route.get('x3'))
            except AttributeError:
                _cp = _vehicle.find('critical_point')
                x10, x20, x30 = float(_cp.get('x1')), float(_cp.get('x2')), float(_cp.get('x3'))
            S, L = [], []
            for _node in _prev_route:
                Sk, Lk = int(_node.get('Sk')), float(_node.get('Lk'))
                S.append(Sk)
                L.append(Lk)
            route = (tuple(S), tuple(L))
            vehicles[ev_id].set_route(route, x10, x20, x30)

        vehicles_to_route.append(ev_id)

    if instance:
        network = net.from_element_tree(tree)
    else:
        network = None

    fleet = Fleet(vehicles, network, tuple(vehicles_to_route))
    return fleet


def routes_from_csv_folder(folder_path: str, fleet: Fleet) -> RouteDict:
    routes: RouteDict = {}
    for ev in fleet.vehicles.values():
        path = f'{folder_path}EV{ev.id}_operation.csv'
        df = pd.read_csv(path, index_col=0)
        Sk, Lk = tuple(df.Sk.values), tuple(df.Lk.values)
        x10, x20, x30 = df.x1_leaving.iloc[0], df.x2_leaving.iloc[0], df.x3_leaving.iloc[0]
        routes[ev.id] = ((Sk, Lk), x10, x20, x30)
    return routes


def routes_from_xml(path: str, fleet: Fleet) -> RouteDict:
    # Open XML file
    tree = ET.parse(path)
    _fleet: ET = tree.find('fleet')
    _fleet = _fleet if _fleet is not None else tree.getroot()

    routes: RouteDict = {}
    for _ev in _fleet:
        _pr = _ev.find('previous_route')
        _cp = _ev.find('critical_point')
        Sk, Lk = tuple(int(node.get('Sk')) for node in _pr), tuple(float(node.get('Lk')) for node in _pr)
        x10, x20, x30 = float(_cp.get('x1')), float(_cp.get('x2')), float(_cp.get('x3'))
        routes[int(_ev.get('id'))] = ((Sk, Lk), x10, x20, x30)
    return routes
