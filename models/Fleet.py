import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Union, NamedTuple
from dataclasses import dataclass

import numpy as np
from numpy import ndarray, zeros
import time

from models.ElectricVehicle import ElectricVehicle
import models.Network as net
from models.Node import DepotNode, CustomerNode, ChargeStationNode
from models.Edge import Edge, DynamicEdge
import res.IOTools

# Visualization tools
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models.annotations import Arrow, Label
from bokeh.models.arrow_heads import VeeHead
from bokeh.models import Whisker, Span, Range1d


# CLASSES
class InitialCondition(NamedTuple):
    S0: int
    L0: float
    x1_0: float
    x2_0: float
    x3_0: float


# TYPES
IndividualType = List
IndicesType = Dict[int, Tuple[int, int]]
StartingPointsType = Dict[int, InitialCondition]
RouteVector = Tuple[Tuple[int, ...], Tuple[float, ...]]
RouteDict = Dict[int, Tuple[RouteVector, float, float, float]]


def distance(results, multi: ndarray, b: ndarray):
    return np.sum([dist_fun(multi[i, 0], b[i, 0]) for i, result in enumerate(results) if not result])


def dist_fun(x, y):
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
    iTheta0 = theta_index
    iTheta1 = theta_index + network_size
    op_vector[iTheta0:iTheta1] = init_state

    counters = [[TupleIndex([0, 0]), 0] for _ in range(len(time_vectors))]

    for k in range(1, events_count):
        iTheta0 += network_size
        iTheta1 += network_size
        min_t, ind_ev = min([(time_vectors[ind][x[0][1]][x[0][0]], ind) for ind, x in enumerate(counters) if not x[1]])

        if counters[ind_ev][0][1]:
            event = 1
        else:
            event = -1

        node = time_vectors[ind_ev][2][counters[ind_ev][0][0] + counters[ind_ev][0][1]]
        op_vector[iTheta0:iTheta1] = op_vector[iTheta0 - network_size:iTheta1 - network_size]
        op_vector[iTheta0 + node] += event
        counters[ind_ev][0] = counters[ind_ev][0] + 1
        if not counters[ind_ev][0][1] and len(time_vectors[ind_ev][2]) - 1 == counters[ind_ev][0][0] + \
                counters[ind_ev][0][1]:
            counters[ind_ev][1] = 1
    return


class TupleIndex(list):
    def __add__(self, other: int):
        i = 2 * self[0] + self[1] + other
        return TupleIndex(integer_to_tuple2(i))


class Fleet:
    vehicles: Dict[int, ElectricVehicle]
    network: net.Network
    vehicles_to_route: Tuple[int, ...]
    theta_vector: Union[ndarray, None]
    optimization_vector: Union[ndarray, None]
    optimization_vector_indices: Union[Tuple, None]
    starting_points: StartingPointsType

    def __init__(self, vehicles=None, network=None, vehicles_to_route=None):
        self.set_vehicles(vehicles)
        self.set_network(network)
        self.set_vehicles_to_route(vehicles_to_route)

        self.theta_vector = None
        self.optimization_vector = None
        self.optimization_vector_indices = None

        self.starting_points = {}

    def set_vehicles(self, vehicles: Dict[int, ElectricVehicle]) -> None:
        self.vehicles = vehicles

    def set_network(self, net: net.Network) -> None:
        self.network = net

    def assign_customers_in_route(self):
        for ev in self.vehicles.values():
            ev.assign_customers_in_route(self.network)

    def set_vehicles_to_route(self, vehicles: List[int]) -> None:
        if vehicles:
            self.vehicles_to_route = tuple(vehicles)

    def set_routes_of_vehicles(self, routes: RouteDict) -> None:
        for id_ev, (route, dep_time, dep_soc, dep_pay) in routes.items():
            self.vehicles[id_ev].set_route(route, dep_time, dep_soc, dep_pay)
            self.vehicles[id_ev].iterate_space(self.network)

    def create_optimization_vector(self) -> ndarray:
        # It is assumed that each EV has a set route by using the ev.set_rout(...) method
        # 0. Preallocate optimization vector
        sum_si = sum([len(self.vehicles[x].route[0]) for x in self.vehicles_to_route])
        length_op_vector = sum_si * (8 + 2 * len(self.network)) - 2 * len(self.vehicles_to_route) * (
                len(self.network) + 1) + len(self.network)
        self.optimization_vector = zeros(length_op_vector)

        # 1. Iterate each ev state to fill their matrices
        iS = 0
        iL = sum_si
        ix1 = iL + sum_si
        ix2 = ix1 + sum_si
        ix3 = ix2 + sum_si
        iD = ix3 + sum_si
        iT = iD + sum_si
        iE = iT + sum_si - len(self.vehicles_to_route)
        iTheta = iE + sum_si - len(self.vehicles_to_route)

        self.optimization_vector_indices = (iS, iL, ix1, ix2, ix3, iD, iT, iE, iTheta)

        t_list = []
        for id_ev in self.vehicles_to_route:
            si = len(self.vehicles[id_ev].route[0])
            self.optimization_vector[iS:iS + si] = self.vehicles[id_ev].route[0]
            self.optimization_vector[iL:iL + si] = self.vehicles[id_ev].route[1]
            self.optimization_vector[ix1:ix1 + si] = self.vehicles[id_ev].state_reaching[0, :]
            self.optimization_vector[ix2:ix2 + si] = self.vehicles[id_ev].state_reaching[1, :]
            self.optimization_vector[ix3:ix3 + si] = self.vehicles[id_ev].state_reaching[2, :]
            self.optimization_vector[iD:iD + si] = self.vehicles[id_ev].charging_times
            self.optimization_vector[iT:iT + si - 1] = self.vehicles[id_ev].travel_times
            self.optimization_vector[iE:iE + si - 1] = self.vehicles[id_ev].energy_consumption

            t_list.append((self.vehicles[id_ev].state_leaving[0, :-1], self.vehicles[id_ev].state_reaching[0, 1:],
                           self.vehicles[id_ev].route[0]))

            iS += si
            iL += si
            ix1 += si
            ix2 += si
            ix3 += si
            iD += si
            iT += si - 1
            iE += si - 1

        # 2. Create theta vector
        init_theta = zeros(len(self.network))
        init_theta[0] = len(self.vehicles)
        theta_vector_in_array(init_theta, t_list, len(self.network), 2 * sum_si - 2 * len(self.vehicles) + 1,
                              self.optimization_vector, iTheta)

        # 3. Create optimization vector
        return self.optimization_vector

    def cost_function(self) -> Tuple:
        iS, iL, ix1, ix2, ix3, iD, iT, iE, iTheta = self.optimization_vector_indices
        op_vector = self.optimization_vector
        cost_tt = np.sum(op_vector[iT:iE])
        cost_ec = np.sum(op_vector[iE:iTheta])
        cost_chg_op = np.sum(op_vector[iD:iT])
        cost_chg_cost = np.sum(op_vector[iL:ix1])
        return cost_tt, cost_ec, cost_chg_op, cost_chg_cost

    def feasible(self) -> (bool, Union[int, float]):
        # Variables to return
        is_feasible = True
        dist = 0

        # Variables from the optimization vector and vehicles
        n_vehicles = len(self.vehicles)
        n_customers = sum([1 for id_ev in self.vehicles_to_route
                           for node in self.vehicles[id_ev].route[0] if self.network.isCustomer(node)])
        network = self.network
        sum_si = np.sum(len(vehicle.route[0]) for _, vehicle in self.vehicles.items())
        length_op_vector = len(self.optimization_vector)

        iS, iL, ix1, ix2, ix3, iD, iT, iE, iTheta = self.optimization_vector_indices

        # Amount of rows
        rows = 0

        rows += n_vehicles  # 2.16
        rows += n_customers  # 2.17
        rows += n_customers  # 2.18
        rows += sum_si  # 2.25-1
        rows += sum_si  # 2.25-2
        rows += sum_si  # 2.26-1
        rows += sum_si  # 2.26-2
        rows += (int(len(self.optimization_vector[iTheta:])/len(self.network.nodes)))*len(self.network.charging_stations)

        # Matrices
        A = np.zeros((rows, length_op_vector))
        b = np.zeros((rows, 1))

        # Start filling
        row = 0

        # 2.16
        si = 0
        for j, vehicle in self.vehicles.items():
            A[row, ix1 + si] = -1.0
            si += len(vehicle.route[0])
            A[row, ix1 + si - 1] = 1.0
            b[row] = vehicle.max_tour_duration
            row += 1

        # 2.17 & 2.18
        si = 0
        for _, vehicle in self.vehicles.items():
            for k, (Sk, Lk) in enumerate(zip(vehicle.route[0], vehicle.route[1])):
                if network.isCustomer(Sk):
                    A[row, ix1 + si + k] = -1.0
                    b[row] = -network.nodes[Sk].timeWindowDown
                    row += 1

                    A[row, ix1 + si + k] = 1.0
                    b[row] = network.nodes[Sk].timeWindowUp - network.spent_time(Sk, None, None)
                    row += 1
            si += len(vehicle.route[0])

        # 2.25-1 & 2.25-2
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

        # 2.26-1 & 2.26-2
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

        # constraint CS capacity
        num_nodes = len(self.network.nodes)
        for i in self.network.charging_stations:
            for k in range(int(len(self.optimization_vector[iTheta:]) / num_nodes)):
                A[row, iTheta + num_nodes*(k+1) - i] = 1.0
                b[row] = self.network.nodes[i].maximumParallelOperations
                row += 1

        # Check
        multi = np.matmul(A, np.vstack(self.optimization_vector.T))
        boolList = multi <= b
        for result in boolList:
            if not result:
                dist = distance(boolList, multi, b)
                is_feasible = False
                break

        return is_feasible, dist

    # Realtime tools
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

    def plot_operation(self):
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
            figX1 = figure(plot_width=600, plot_height=450,
                           title=f'Time the vehicle {id_ev} leaves',
                           toolbar_location='right')
            figX2 = figure(plot_width=600, plot_height=450,
                           title=f'SOC evolution vehicle {id_ev}',
                           y_range=(0, 100),
                           toolbar_location='right')
            figX3 = figure(plot_width=600, plot_height=450,
                           title=f'Payload evolution vehicle {id_ev}',
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
                    tWindowsCenter = (node_instance.timeWindowUp + node_instance.timeWindowDown) / 2.0
                    tWindowsWidth = (node_instance.timeWindowUp - node_instance.timeWindowDown) / 2.0
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
            figX1.xaxis.axis_label = 'k'
            figX1.yaxis.axis_label = 'Time of the day (min)'
            figX1.axis.axis_label_text_font_size = '15pt'
            figX1.axis.major_label_text_font_size = '13pt'
            figX1.title.text_font_size = '15pt'

            figX2.xaxis.axis_label = 'k'
            figX2.yaxis.axis_label = 'SOC (%)'
            figX2.axis.axis_label_text_font_size = '15pt'
            figX2.axis.major_label_text_font_size = '13pt'
            figX2.title.text_font_size = '15pt'

            figX3.xaxis.axis_label = 'k'
            figX3.yaxis.axis_label = 'Payload (ton)'
            figX3.axis.axis_label_text_font_size = '15pt'
            figX3.axis.major_label_text_font_size = '13pt'
            figX3.title.text_font_size = '15pt'

            # Show plots of vehicles
            show(figX1)
            time.sleep(0.5)
            show(figX2)
            time.sleep(0.5)
            show(figX3)
            time.sleep(0.5)

        # Now, charging stations
        network_size = len(self.network)
        full_operation_theta_vector = self.optimization_vector[self.optimization_vector_indices[8]:]
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
            fig.axis.axis_label_text_font_size = '15pt'
            fig.axis.major_label_text_font_size = '13pt'
            fig.title.text_font_size = '15pt'
            show(fig)
            time.sleep(0.5)
        return

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
            res.IOTools.write_pretty_xml(path)
        return tree


# A tool to import a fleet from an XML file
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
            max_payload = float(_vehicle.get('max_payload'))

            vehicles[ev_id] = ElectricVehicle(ev_id, alpha_up, alpha_down, max_tour_duration, battery_capacity,
                                              max_payload)
            if assign_customers:
                _assigned_customers = _vehicle.find('assigned_customers')
                assigned_customers = tuple(int(x.get('id')) for x in _assigned_customers)
                vehicles[ev_id].set_customers_to_visit(assigned_customers)

            vehicles_to_route.append(ev_id)

    fleet = Fleet(vehicles, network, tuple(vehicles_to_route))

    return fleet
