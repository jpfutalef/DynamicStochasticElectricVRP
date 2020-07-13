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


def dist_fun(x: ndarray, y: ndarray):
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


class Fleet:
    vehicles: Dict[int, ElectricVehicle]
    network: net.Network
    vehicles_to_route: Tuple[int, ...]
    theta_vector: Union[np.ndarray, None]
    optimization_vector: Union[np.ndarray, None]
    optimization_vector_indices: Union[Tuple, None]
    starting_points: Dict[int, InitialCondition]
    eta_table: np.ndarray
    eta_model: NearestNeighbors

    def __init__(self, vehicles=None, network=None, vehicles_to_route=None):
        self.set_vehicles(vehicles)
        self.set_network(network)
        self.set_vehicles_to_route(vehicles_to_route)

        self.theta_vector = None
        self.optimization_vector = None
        self.optimization_vector_indices = None

        self.starting_points = {}

        self.eta_table = np.array([[0.500, 1.00, 0.999930],
                                   [0.625, 0.75, 0.999963],
                                   [0.375, 0.75, 0.999972],
                                   [0.750, 0.50, 0.999981],
                                   [0.500, 0.50, 0.999987],
                                   [0.250, 0.50, 0.999991],
                                   [0.875, 0.25, 0.999997],
                                   [0.625, 0.25, 0.999996],
                                   [0.500, 0.25, 0.999996],
                                   [0.375, 0.25, 0.999981],
                                   [0.125, 0.25, 0.999987]])
        self.eta_model = NearestNeighbors(n_neighbors=3).fit(self.eta_table[:, 0:2])

    def __len__(self):
        return len(self.vehicles)

    def set_vehicles(self, vehicles: Dict[int, ElectricVehicle]) -> None:
        self.vehicles = vehicles

    def set_network(self, network: Network) -> None:
        self.network = network

    def resize_fleet(self, new_size, as_new=True, based_on=0):
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

    def drop_vehicle(self, ev_id: int) -> None:
        del self.vehicles[ev_id]

    def assign_customers_in_route(self):
        for ev in self.vehicles.values():
            ev.assign_customers_in_route(self.network)

    def set_vehicles_to_route(self, vehicles: List[int]) -> None:
        if vehicles:
            self.vehicles_to_route = tuple(vehicles)

    def set_routes_of_vehicles(self, routes: RouteDict, iterate=True) -> None:
        for id_ev, (route, dep_time, dep_soc, dep_pay) in routes.items():
            self.vehicles[id_ev].set_route(route, dep_time, dep_soc, dep_pay)
            if iterate:
                self.vehicles[id_ev].step(self.network)

    def set_eta(self, etas: Dict[int, float]):
        for id_ev, ev in self.vehicles.items():
            ev.eta0 = etas[id_ev]

    def create_optimization_vector(self) -> np.ndarray:
        # It is assumed that each EV has a set route by using the ev.set_rout(...) method
        # 0. Preallocate optimization vector
        sum_si = sum([len(self.vehicles[x].route[0]) for x in self.vehicles_to_route])
        length_op_vector = sum_si * (10 + 2 * len(self.network)) - 2 * len(self.vehicles_to_route) * (
                len(self.network) + 1) + len(self.network)
        self.optimization_vector = np.zeros(length_op_vector)

        # 1. Iterate each ev state to fill their matrices
        iS = 0
        iL = sum_si
        ix1 = iL + sum_si
        ix2 = ix1 + sum_si
        ix3 = ix2 + sum_si
        ix1L = ix3 + sum_si
        iWT = ix1L + sum_si
        iD = iWT + sum_si
        iT = iD + sum_si
        iE = iT + sum_si - len(self.vehicles_to_route)
        iTheta = iE + sum_si - len(self.vehicles_to_route)

        self.optimization_vector_indices = (iS, iL, ix1, ix2, ix3, ix1L, iWT, iD, iT, iE, iTheta)

        time_vectors = []
        for id_ev in self.vehicles_to_route:
            si = len(self.vehicles[id_ev].route[0])
            self.optimization_vector[iS:iS + si] = self.vehicles[id_ev].route[0]
            self.optimization_vector[iL:iL + si] = self.vehicles[id_ev].route[1]
            self.optimization_vector[ix1:ix1 + si] = self.vehicles[id_ev].state_reaching[0, :]
            self.optimization_vector[ix2:ix2 + si] = self.vehicles[id_ev].state_reaching[1, :]
            self.optimization_vector[ix3:ix3 + si] = self.vehicles[id_ev].state_reaching[2, :]
            self.optimization_vector[ix1L:ix1L + si] = self.vehicles[id_ev].state_leaving[0, :]
            self.optimization_vector[iWT:iWT + si] = self.vehicles[id_ev].waiting_times
            self.optimization_vector[iD:iD + si] = self.vehicles[id_ev].charging_times
            self.optimization_vector[iT:iT + si - 1] = self.vehicles[id_ev].travel_times
            self.optimization_vector[iE:iE + si - 1] = self.vehicles[id_ev].energy_consumption

            time_vectors.append((self.vehicles[id_ev].state_leaving[0, :-1], self.vehicles[id_ev].state_reaching[0, 1:],
                                 self.vehicles[id_ev].route[0]))

            iS += si
            iL += si
            ix1 += si
            ix2 += si
            ix3 += si
            ix1L += si
            iWT += si
            iD += si
            iT += si - 1
            iE += si - 1

        # 2. Create theta vector
        init_theta = np.zeros(len(self.network))
        init_theta[0] = len(self.vehicles)
        theta_vector_in_array(init_theta, time_vectors, len(self.network), 2 * sum_si - 2 * len(self.vehicles) + 1,
                              self.optimization_vector, iTheta)

        # 3. Create optimization vector
        return self.optimization_vector

    def cost_function(self) -> Tuple:
        iS, iL, ix1, ix2, ix3, ix1L, iWT, iD, iT, iE, iTheta = self.optimization_vector_indices
        op_vector = self.optimization_vector
        cost_tt = np.sum(op_vector[iT:iE])
        cost_ec = np.sum(op_vector[iE:iTheta])
        cost_chg_op = np.sum(op_vector[iD:iT])
        cost_chg_cost = np.sum(op_vector[iL:ix1])
        cost_wait_time = np.sum(op_vector[iWT:iD])
        return cost_tt, cost_ec, cost_chg_op, cost_chg_cost, cost_wait_time

    def feasible(self) -> (bool, Union[int, float]):
        # 1. Variables to return
        is_feasible = True
        dist = 0

        # 2. Variables from the optimization vector and vehicles
        n_vehicles = len(self.vehicles)
        n_customers = sum([1 for id_ev in self.vehicles_to_route
                           for node in self.vehicles[id_ev].route[0] if self.network.isCustomer(node)])
        network = self.network
        sum_si = np.sum(len(vehicle.route[0]) for _, vehicle in self.vehicles.items())
        length_op_vector = len(self.optimization_vector)

        iS, iL, ix1, ix2, ix3, ix1L, iWT, iD, iT, iE, iTheta = self.optimization_vector_indices

        # 3. Amount of rows
        rows = 0

        rows += n_vehicles  # Constraint 1 - Maximum tour time
        rows += n_vehicles  # Constraint 2 - Maximum weight
        rows += 2 * n_customers  # Constraint 3 & 4 - Time window up and time window down
        rows += 2 * sum_si  # Constraint 5 & 6 - SOH policies when EV arrives
        rows += 2 * sum_si  # Constraint 7 & 8 - SOH policies when EV leaves
        rows += (int(len(self.optimization_vector[iTheta:]) / len(self.network.nodes))) * len(
            self.network.charging_stations)  # Constraint 9 - CS capacities

        # 4. Matrices
        A = np.zeros((rows, length_op_vector))
        b = np.zeros(rows)

        # 5. Start filling
        row = 0

        # Constraint 1 - Maximum tour time
        si = 0
        for j, vehicle in self.vehicles.items():
            A[row, ix1 + si] = -1.0
            si += len(vehicle.route[0])
            A[row, ix1 + si - 1] = 1.0
            b[row] = vehicle.max_tour_duration
            row += 1

        # Constraint 2 - Maximum weight
        si = 0
        for j, vehicle in self.vehicles.items():
            A[row, ix3 + si] = 1.0
            b[row] = vehicle.max_payload
            si += len(vehicle.route[0])
            row += 1

        # Constraint 3 & 4 - Time window up and time window down
        si = 0
        for _, vehicle in self.vehicles.items():
            for k, (Sk, Lk) in enumerate(zip(vehicle.route[0], vehicle.route[1])):
                if network.isCustomer(Sk):
                    A[row, ix1 + si + k] = -1.0
                    b[row] = -network.nodes[Sk].time_window_low
                    row += 1

                    A[row, ix1L + si + k] = 1.0
                    b[row] = network.nodes[Sk].time_window_upp
                    row += 1
            si += len(vehicle.route[0])

        # Constraint 5 & 6 - SOH policies when EV arrives
        si = 0
        for _, vehicle in self.vehicles.items():
            for k, (Sk, Lk) in enumerate(zip(vehicle.route[0], vehicle.route[1])):
                A[row, ix2 + si + k] = -1.0
                b[row] = -vehicle.alpha_down
                row += 1

                A[row, ix2 + si + k] = 1.0
                b[row] = vehicle.alpha_up
                row += 1
            si += len(vehicle.route[0])

        # Constraint 7 & 8 - SOH policies when EV leaves
        si = 0
        for _, vehicle in self.vehicles.items():
            for k, (Sk, Lk) in enumerate(zip(vehicle.route[0], vehicle.route[1])):
                A[row, ix2 + si + k] = -1.0
                b[row] = -vehicle.alpha_down + Lk
                row += 1

                A[row, ix2 + si + k] = 1.0
                b[row] = vehicle.alpha_up - Lk
                row += 1
            si += len(vehicle.route[0])

        # Constraint 9 - CS capacities
        num_nodes = len(self.network.nodes)
        for i in self.network.charging_stations:
            for k in range(int(len(self.optimization_vector[iTheta:]) / num_nodes)):
                A[row, iTheta + num_nodes * (k + 1) - i] = 1.0
                b[row] = self.network.nodes[i].capacity
                row += 1

        # 6. Check
        multi = np.matmul(A, self.optimization_vector)
        boolList = multi <= b
        for result in boolList:
            if not result:
                dist = distance(boolList, multi, b)
                is_feasible = False
                break
        return is_feasible, dist

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
                self.vehicles[ev_id].set_route(route_from_crit_point, x1_0, x2_0, x3_0, stochastic=False)

        self.set_vehicles_to_route(vehicles_to_route)

        return tree

    def plot_operation_pyplot(self, arrow_colors=('SteelBlue', 'Crimson', 'SeaGreen'), fig_size=(16, 5),
                              label_offset=(.15, -6), subplots=True, save_to=None):
        figs = []
        for id_ev, vehicle in self.vehicles.items():
            Si, Li = vehicle.route[0], vehicle.route[1]
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
            plt.errorbar(X_tw, Y_tw, tw_sigma, ecolor='black', fmt='none', capsize=6, elinewidth=1, zorder=5)

            # time in nodes customers
            X_node = [i for i in range(si) if self.network.isCustomer(Si[i])]
            Y_node = [t for i, t in enumerate(r_time) if self.network.isCustomer(Si[i])]
            U_node = [0 for i in range(si) if self.network.isCustomer(Si[i])]
            V_node = [l_time[i] - r_time[i] for i in range(si) if self.network.isCustomer(Si[i])]
            plt.quiver(X_node, Y_node, U_node, V_node, scale=1, angles='xy', scale_units='xy',
                       color=arrow_colors[2], width=0.0004 * fig_size[0], headwidth=6, zorder=10)

            # time in nodes CS
            X_node = [i for i in range(si) if self.network.isChargingStation(Si[i])]
            Y_node = [t for i, t in enumerate(r_time) if self.network.isChargingStation(Si[i])]
            U_node = [0 for i in range(si) if self.network.isChargingStation(Si[i])]
            V_node = [l_time[i] - r_time[i] for i in range(si) if self.network.isChargingStation(Si[i])]
            plt.quiver(X_node, Y_node, U_node, V_node, scale=1, angles='xy', scale_units='xy',
                       color=arrow_colors[1], width=0.0004 * fig_size[0], headwidth=6, zorder=10)

            # travel time
            X_edge, Y_edge = range(si - 1), l_time[0:-1]
            U_edge, V_edge = np.ones(si - 1), r_time[1:] - l_time[0:-1]
            color_edge = [arrow_colors[0]] * si
            plt.quiver(X_edge, Y_edge, U_edge, V_edge, scale=1, angles='xy', scale_units='xy', color=color_edge,
                       width=0.0004 * fig_size[0], zorder=15)

            # Annotate nodes
            labels = [str(i) for i in vehicle.route[0]]
            pos = [(x, y) for x, y in zip(range(si), r_time)]
            for label, (x, y) in zip(labels, pos):
                plt.annotate(label, (x + label_offset[0], y + label_offset[1]))

            plt.legend(('Serving customer', 'Charging operation', 'Travelling', 'Cust. time window'), fontsize='small')
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
        return figs

    def draw_operation(self, color_route='red', **kwargs):
        nodes_id = {i: i for i, node in self.network.nodes.items()}
        fig, g = self.network.draw(labels=nodes_id, **kwargs)
        pos = {i: (node.pos_x, node.pos_y) for i, node in self.network.nodes.items()}
        cc = 0
        for id_ev, ev in self.vehicles.items():
            edges = [(Sk0, Sk1) for Sk0, Sk1 in zip(ev.route[0][0:-1], ev.route[0][1:])]
            if type(color_route) == str:
                nx.draw_networkx_edges(g, pos, edgelist=edges, ax=fig.get_axes()[0], edge_color=color_route)
            else:
                nx.draw_networkx_edges(g, pos, edgelist=edges, ax=fig.get_axes()[0], edge_color=color_route[cc])
                cc += 1
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

    def xml_tree(self, assign_customers=False, with_routes=False):
        _fleet = ET.Element('fleet')
        ev_id = 0
        for vehicle in self.vehicles.values():
            if assign_customers:
                if len(vehicle.assigned_customers) > 0:
                    _fleet.append(vehicle.xml_element(assign_customers, with_routes, ev_id))
                    ev_id += 1
            else:
                _fleet.append(vehicle.xml_element(assign_customers, with_routes, ev_id))
        return _fleet

    def write_xml(self, path, network_in_file=False, assign_customers=False, with_routes=False,
                  print_pretty=False):
        tree = self.xml_tree(assign_customers, with_routes)
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


def from_xml(path, assign_customers=False):
    # Open XML file
    tree = ET.parse(path)
    _info: ET = tree.find('info')
    _network: ET = tree.find('network')
    _fleet: ET = tree.find('fleet')
    _technologies: ET = _network.find('technologies')

    # Network data
    network = net.from_element_tree(tree)

    # Fleet data
    vehicles = {}
    vehicles_to_route = []
    for _vehicle in _fleet:
        add_ev = True

        _cp = _vehicle.find('critical_point')
        if _cp is not None and int(_cp.get('k')) == -1:
            add_ev = False

        if add_ev:
            ev_id = int(_vehicle.get('id'))
            max_tour_duration = float(_vehicle.get('max_tour_duration'))
            alpha_up = float(_vehicle.get('alpha_up'))
            alpha_down = float(_vehicle.get('alpha_down'))
            battery_capacity = float(_vehicle.get('battery_capacity'))
            battery_capacity_nominal = float(_vehicle.get('battery_capacity_nominal'))
            max_payload = float(_vehicle.get('max_payload'))
            weight = float(_vehicle.get('weight'))
            vehicles[ev_id] = ElectricVehicle(ev_id, weight, battery_capacity,battery_capacity_nominal, alpha_up,
                                              alpha_down, max_tour_duration, max_payload)
            if assign_customers:
                _assigned_customers = _vehicle.find('assigned_customers')
                assigned_customers = tuple(int(x.get('id')) for x in _assigned_customers)
                vehicles[ev_id].set_customers_to_visit(assigned_customers)

            vehicles_to_route.append(ev_id)

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

    routes: RouteDict = {}
    for _ev in _fleet:
        _pr = _ev.find('previous_route')
        _cp = _ev.find('critical_point')
        Sk, Lk = tuple(int(node.get('Sk')) for node in _pr), tuple(float(node.get('Lk')) for node in _pr)
        x10, x20, x30 = float(_cp.get('x1')), float(_cp.get('x2')), float(_cp.get('x3'))
        routes[int(_ev.get('id'))] = ((Sk, Lk), x10, x20, x30)
    return routes
