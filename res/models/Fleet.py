import xml.etree.ElementTree as ET
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Dict, Union, List, Type
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import res.models.ElectricVehicle as EV
import res.models.Network as Network
import res.tools.IOTools as IOTools
from res.models.Penalization import penalization_deterministic, penalization_stochastic, normal_cdf

'''
MAIN FLEET DEFINITION
'''


@dataclass
class Fleet:
    vehicles: Dict[int, EV.ElectricVehicle]
    network: Network.DeterministicCapacitatedNetwork = None
    vehicles_to_route: Tuple[int, ...] = None
    default_init_theta: np.ndarray = None
    hard_penalization: Union[int, float] = 0.0
    type: str = None

    def __post_init__(self):
        self.set_vehicles_to_route(self.vehicles_to_route)
        self.set_network(self.network)
        self.type = self.__class__.__name__

    def __len__(self):
        return len(self.vehicles)

    def reset(self):
        return

    def set_vehicles(self, vehicles: Dict[int, EV.ElectricVehicle]):
        self.vehicles = vehicles

    def set_network(self, network: Network = None):
        self.network = network
        self.set_default_init_theta()

    def set_default_init_theta(self, init_theta: np.ndarray = None):
        if self.network:
            self.default_init_theta = init_theta if init_theta else np.array(
                [len(self)] + [0 for _ in range(len(self.network) - 1)])

    def set_vehicles_to_route(self, vehicles: Union[List[int], Tuple[int, ...]] = None):
        """
        Set which vehicles consider for routing.
        @param vehicles: List or tuple of vehicle IDs, indicating which ones to route. If None (default) passed, all
        EVs are considered.
        @return: None
        """
        if vehicles:
            self.vehicles_to_route = tuple(vehicles)
        else:
            self.vehicles_to_route = tuple(self.vehicles.keys())

    def drop_vehicle_to_route(self, ev_id: int):
        to_route = self.vehicles_to_route
        i = to_route.index(ev_id) if ev_id in to_route else None
        self.vehicles_to_route = to_route[:i] + to_route[i + 1:] if i else to_route

    def remove_vehicle_from_fleet(self, ev_id: int):
        if ev_id in self.vehicles.keys():
            del self.vehicles[ev_id]
            self.drop_vehicle_to_route(ev_id)

    def resize_fleet(self, new_size, based_on=None):
        ev_base = deepcopy(based_on) if based_on else deepcopy(self.vehicles[0])
        ev_base.reset(self.network)
        self.vehicles = {}
        for i in range(new_size):
            ev = deepcopy(ev_base)
            ev.id = i
            self.vehicles[i] = ev
        self.set_vehicles_to_route()
        self.set_default_init_theta()

    def new_soc_policy(self, alpha_down, alpha_upp):
        for ev in self.vehicles.values():
            ev.alpha_down = alpha_down
            ev.alpha_up = alpha_upp

    def relax_time_windows(self):
        for customer in self.network.customers:
            self.network.nodes[customer].time_window_low = -np.infty
            self.network.nodes[customer].time_window_upp = np.infty

    def modify_cs_capacities(self, new_capacity: Union[int, Mapping]):
        if type(new_capacity) is int:
            for charging_station in self.network.charging_stations:
                self.network.nodes[charging_station].capacity = new_capacity
        else:
            if len(new_capacity) != len(self.network.charging_stations):
                raise ValueError("Number of passed CS capacities must match number of CSs in the network.")
            for charging_station, capacity in zip(self.network.charging_stations, new_capacity):
                self.network.nodes[charging_station].capacity = capacity

    def assign_customers_in_route(self):
        [ev.assign_customers_in_route(self.network) for ev in self.vehicles.values()]

    def set_vehicle_route(self, ev_id: int, S: Tuple[int, ...], L: Tuple[float, ...], x1_0: float, x2_0: float,
                          x3_0: float):
        self.vehicles[ev_id].set_route(S, L, x1_0, x2_0, x3_0)

    def set_routes_of_vehicles(self, routes: Dict[int, Tuple[Tuple[int, ...], Tuple[float, ...], float, float, float]]):
        for id_ev, (S, L, x1_0, x2_0, x3_0) in routes.items():
            self.set_vehicle_route(id_ev, S, L, x1_0, x2_0, x3_0)

    def iterate(self, init_theta: np.ndarray = None):
        for ev in self.vehicles.values():
            ev.step(self.network)
        time_vectors = []
        for id_ev in self.vehicles_to_route:
            ev = self.vehicles[id_ev]
            time_vectors.append((ev.state_leaving[0, :-1], ev.state_reaching[0, 1:], ev.S))
        if init_theta is not None:
            self.network.iterate_cs_capacities(init_theta, time_vectors, len(self))
        else:
            self.network.iterate_cs_capacities(self.default_init_theta, time_vectors, len(self))

    def cost_function(self):
        cost_tt = sum([sum(self.vehicles[id_ev].travel_times) for id_ev in self.vehicles_to_route])
        cost_ec = sum([sum(self.vehicles[id_ev].energy_consumption) for id_ev in self.vehicles_to_route])
        cost_chg_time = sum([sum(self.vehicles[id_ev].charging_times) for id_ev in self.vehicles_to_route])

        cost_chg_cost = 0.
        for id_ev in self.vehicles_to_route:
            ev = self.vehicles[id_ev]
            for (Sk, Lk) in zip(ev.S, ev.L):
                if self.network.is_charging_station(Sk):
                    price = self.network.nodes[Sk].price
                    cost_chg_cost += price * Lk * ev.battery_capacity

        return cost_tt, cost_ec, cost_chg_time, cost_chg_cost

    def feasible(self) -> Tuple[bool, Union[int, float], bool]:
        return self.check_feasibility()

    def check_feasibility(self) -> Tuple[bool, Union[int, float], bool]:
        # 1. Variables to return
        is_feasible = True
        penalization = 0

        # 2. Check feasibility for each vehicle
        for ev in self.vehicles.values():
            penalization += ev.check_feasibility(self.network)

        # 3. Check feasibility for CS capacities
        penalization += self.network.check_feasibility()

        # If penalization exists, solution is not feasible
        if penalization > 0:
            penalization += self.hard_penalization
            is_feasible = False
        accept = is_feasible

        return is_feasible, penalization, accept

    def xml_element(self, assign_customers=False, with_routes=False):
        _fleet = ET.Element('fleet')
        _fleet.set('type', self.type)
        _fleet.set('hard_penalization', str(self.hard_penalization))
        for vehicle in self.vehicles.values():
            _fleet.append(vehicle.xml_element(assign_customers, with_routes))
        return _fleet

    def write_xml(self, filepath: Path, network_in_file=False, assign_customers=False, with_routes=False,
                  print_pretty=False):
        tree = self.xml_element(assign_customers, with_routes)
        if network_in_file and self.network is not None:
            instance_tree = ET.Element('instance')
            instance_tree.append(self.network.xml_tree())
            instance_tree.append(tree)
            tree = instance_tree

        if print_pretty:
            IOTools.write_pretty_xml(filepath, tree)
        else:
            ET.ElementTree(tree).write(filepath)

    @classmethod
    def from_xml_element(cls, element: ET.Element):
        hard_penalization = float(element.get('hard_penalization'))
        vehicles = {}
        for _vehicle in element:
            ev = EV.from_xml_element(_vehicle)
            vehicles[ev.id] = ev
        return cls(vehicles, hard_penalization=hard_penalization)

    @classmethod
    def from_xml(cls, xml_file: Union[str, ET.Element], return_etree=False):
        if type(xml_file) == str:
            tree = ET.parse(xml_file).getroot()
        else:
            tree = xml_file
        is_instance = True if tree.tag == 'instance' else False
        _fleet = tree.find('fleet') if is_instance else tree

        fleet = cls.from_xml_element(_fleet)

        if is_instance:
            _network = tree.find('network')
            network = Network.from_xml_element(_network)
            fleet.set_network(network)

        if return_etree:
            return fleet, tree

        return fleet

    def plot_operation_pyplot(self, size_ev_op=(16, 5), size_cs_occupation=(16, 5), **kwargs):
        figs = []
        figs += self.plot_ev_operation(fig_size=size_ev_op)
        figs.append(self.plot_cs_occupation(figsize=size_cs_occupation))
        return figs

    def plot_ev_operation(self, arrow_colors=('SteelBlue', 'Crimson', 'SeaGreen'), fig_size=(16, 5),
                          label_offset=(.15, -6), subplots=True, save_to=None, arrival_at_depot_times: Dict = None):
        figs = []
        for id_ev, vehicle in self.vehicles.items():
            Si, Li = vehicle.S, vehicle.L
            st_reaching, st_leaving = vehicle.state_reaching, vehicle.state_leaving
            r_time, r_soc, r_payload = st_reaching[0, :], st_reaching[1, :], st_reaching[2, :]
            l_time, l_soc, l_payload = st_leaving[0, :], st_leaving[1, :], st_leaving[2, :]

            si = len(Si)

            fig = plt.figure(figsize=fig_size)

            ### FIG X1 ###
            plt.subplot(131)
            plt.grid()

            # Covariance
            try:
                Y_reaching_std = np.array(
                    [np.sqrt(q) for i, q in enumerate(vehicle.state_reaching_covariance[0, :]) if i % 3 == 0])
                plt.fill_between(range(si), r_time - 3 * Y_reaching_std, r_time + 3 * Y_reaching_std, color='red',
                                 alpha=0.2)

                Y_leaving_std = np.array(
                    [np.sqrt(q) for i, q in enumerate(vehicle.state_reaching_covariance[0, :]) if i % 3 == 0])
                plt.fill_between(range(si), l_time - 3 * Y_leaving_std, l_time + 3 * Y_leaving_std, color='green',
                                 alpha=0.2)
            except AttributeError:
                pass

            # time windows
            X_tw = [k for k, Sk in enumerate(Si) if self.network.is_customer(Sk)]
            Y_tw = [(self.network.nodes[Sk].time_window_upp + self.network.nodes[Sk].time_window_low) / 2.0
                    for k, Sk in enumerate(Si) if self.network.is_customer(Sk)]
            tw_sigma = [(self.network.nodes[Sk].time_window_upp - self.network.nodes[Sk].time_window_low) / 2.0
                        for k, Sk in enumerate(Si) if self.network.is_customer(Sk)]
            plt.errorbar(X_tw, Y_tw, tw_sigma, ecolor='black', fmt='none', capsize=6, elinewidth=1, zorder=5,
                         label='Customer time window')

            # time in nodes customers
            X_node = [i for i in range(si) if self.network.is_customer(Si[i])]
            Y_node = [t for i, t in enumerate(r_time) if self.network.is_customer(Si[i])]
            U_node = [0 for i in range(si) if self.network.is_customer(Si[i])]
            V_node = [l_time[i] - r_time[i] for i in range(si) if self.network.is_customer(Si[i])]
            plt.quiver(X_node, Y_node, U_node, V_node, scale=1, angles='xy', scale_units='xy',
                       color=arrow_colors[2], width=0.0004 * fig_size[0], headwidth=6, zorder=10,
                       label='Serving customer')

            # time in nodes CS
            X_node = [i for i in range(si) if self.network.is_charging_station(Si[i])]
            Y_node = [t for i, t in enumerate(r_time) if self.network.is_charging_station(Si[i])]
            U_node = [0 for i in range(si) if self.network.is_charging_station(Si[i])]
            V_node = [l_time[i] - r_time[i] for i in range(si) if self.network.is_charging_station(Si[i])]
            plt.quiver(X_node, Y_node, U_node, V_node, scale=1, angles='xy', scale_units='xy',
                       color=arrow_colors[1], width=0.0004 * fig_size[0], headwidth=6, zorder=10,
                       label='Charging operation')

            # travel time
            X_edge, Y_edge = range(si - 1), l_time[0:-1]
            U_edge, V_edge = np.ones(si - 1), r_time[1:] - l_time[0:-1]
            color_edge = [arrow_colors[0]] * si
            plt.quiver(X_edge, Y_edge, U_edge, V_edge, scale=1, angles='xy', scale_units='xy', color=color_edge,
                       width=0.0004 * fig_size[0], zorder=15, label='Travelling')

            # Maximum tour time
            if arrival_at_depot_times is not None:
                plt.axhline(arrival_at_depot_times[id_ev], linestyle='--', color='black', label='Maximum tour time')
            else:
                plt.axhline(vehicle.state_reaching[0, 0] + vehicle.current_max_tour_duration, linestyle='--',
                            color='black', label='Maximum tour time')

            # Annotate nodes
            labels = [str(i) for i in vehicle.S]
            pos = [(x, y) for x, y in zip(range(si), r_time)]
            for label, (x, y) in zip(labels, pos):
                plt.annotate(label, (x + label_offset[0], y + label_offset[1]))

            plt.legend(fontsize='small')
            plt.xlabel('Stop')
            plt.ylabel('Time of the day [s]')
            plt.title(f'Arrival and departure times (EV {id_ev})')
            plt.xlim(-.5, si - .5)

            ### FIG X2 ###
            plt.subplot(132)
            plt.grid()

            # Covariance
            try:
                Y_reaching_std = np.array(
                    [np.sqrt(q) for i, q in enumerate(vehicle.state_reaching_covariance[1, 1:]) if i % 3 == 0])
                plt.fill_between(range(si), r_soc - 2 * Y_reaching_std, r_soc + 3 * Y_reaching_std, color='red',
                                 alpha=0.2)

                Y_leaving_std = np.array(
                    [np.sqrt(q) for i, q in enumerate(vehicle.state_reaching_covariance[1, 1:]) if i % 3 == 0])
                plt.fill_between(range(si), l_soc - 2 * Y_leaving_std, l_soc + 3 * Y_leaving_std, color='green',
                                 alpha=0.2)
            except AttributeError:
                pass

            # SOC in Customer
            X_node = [i for i in range(si) if self.network.is_customer(Si[i])]
            Y_node = [t for i, t in enumerate(r_soc) if self.network.is_customer(Si[i])]
            U_node = [0 for i in range(si) if self.network.is_customer(Si[i])]
            V_node = [l_soc[i] - r_soc[i] for i in range(si) if self.network.is_customer(Si[i])]
            plt.quiver(X_node, Y_node, U_node, V_node, scale=1, angles='xy', scale_units='xy',
                       color=arrow_colors[2], width=0.0004 * fig_size[0], headwidth=6, zorder=10,
                       label='Serving customer')

            # SOC in CS
            X_node = [i for i in range(si) if self.network.is_charging_station(Si[i])]
            Y_node = [t for i, t in enumerate(r_soc) if self.network.is_charging_station(Si[i])]
            U_node = [0 for i in range(si) if self.network.is_charging_station(Si[i])]
            V_node = [l_soc[i] - r_soc[i] for i in range(si) if self.network.is_charging_station(Si[i])]
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
            labels = [str(i) for i in vehicle.S]
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
            plt.ylabel('State Of Charge [%]')
            plt.title(f'EV Battery SOC (EV {id_ev})')

            ### FIG X3 ###
            plt.subplot(133)
            plt.grid()
            # payload in nodes
            X_node, Y_node = range(si), r_payload
            U_node, V_node = np.zeros(si), l_payload - r_payload
            color_node = [arrow_colors[2] if self.network.is_customer(i) or self.network.is_depot(i)
                          else arrow_colors[1] for i in Si]
            plt.quiver(X_node, Y_node, U_node, V_node, scale=1, angles='xy', scale_units='xy', color=color_node,
                       width=0.0004 * fig_size[0], headwidth=6, zorder=10, label='Serving customer')

            # traveling payload
            X_edge, Y_edge = range(si - 1), l_payload[0:-1]
            U_edge, V_edge = np.ones(si - 1), r_payload[1:] - l_payload[0:-1]
            color_edge = [arrow_colors[0]] * si
            plt.quiver(X_edge, Y_edge, U_edge, V_edge, scale=1, angles='xy', scale_units='xy', color=color_edge,
                       width=0.0004 * fig_size[0], zorder=15, label='Travelling')

            # Annotate nodes
            labels = [str(i) for i in vehicle.S]
            pos = [(x, y) for x, y in zip(range(si), r_payload)]
            for label, (x, y) in zip(labels, pos):
                plt.annotate(label, (x + label_offset[0], y + label_offset[1]))

            # Max weight
            plt.axhline(vehicle.max_payload, linestyle='--', color='black', label='Max. payload')

            plt.xlabel('Stop')
            plt.ylabel('Payload [kg]')
            plt.title(f' (Payload evolution (EV {id_ev})')
            plt.legend(fontsize='small')

            if save_to is not None:
                fig.savefig(f'{save_to}operation_EV{id_ev}_full.pdf', format='pdf')

            figs.append(fig)

        return figs

    def plot_cs_occupation(self, figsize=(16, 5)):
        # Charging stations occupation
        num_cs = len(self.network.charging_stations)
        fig = plt.figure(figsize=figsize)
        x = range(len(self.network.theta_matrix[0, :]))
        plt.step(x, self.network.theta_matrix[-num_cs:, :].T)
        plt.title('CS Occupation')
        plt.xlabel('Event')
        plt.ylabel('Number of EVs')
        plt.legend(tuple(f'CS {i}' for i in self.network.charging_stations))
        return fig

    def draw_operation(self, color_route='red', **kwargs):
        fig, g = self.network.draw(**kwargs)
        pos = {i: (node.pos_x, node.pos_y) for i, node in self.network.nodes.items()}
        cc = 0
        for id_ev, ev in self.vehicles.items():
            edges = [(Sk0, Sk1) for Sk0, Sk1 in zip(ev.S[0:-1], ev.S[1:])]
            if type(color_route) == str:
                nx.draw_networkx_edges(g, pos, edgelist=edges, ax=fig.get_axes()[0], edge_color=color_route)
            else:
                nx.draw_networkx_edges(g, pos, edgelist=edges, ax=fig.get_axes()[0], edge_color=color_route[cc])
                cc = cc + 1 if cc + 1 < len(color_route) else 0
        return fig, g


'''
GAUSSIAN FLEET DEFINITION
'''


class GaussianFleet(Fleet):
    vehicles: Dict[int, EV.GaussianElectricVehicle]
    network: Network.GaussianCapacitatedNetwork

    def __post_init__(self):
        super(GaussianFleet, self).__post_init__()

    def iterate(self, **kwargs):
        for ev in self.vehicles.values():
            ev.step(self.network)

        self.network.reset_containers()
        iteration_info = {ev.id: (ev.state_leaving[0, :], ev.state_reaching[0, :], ev.state_reaching_covariance,
                                  ev.S) for ev in self.vehicles.values()}
        self.network.fill_deterministics_occupation(iteration_info, **kwargs)
        return

    def set_network(self, network: Network = None):
        self.network = network
        if network:
            self.network.setup_cs_capacities_combinations(len(self))

    def resize_fleet(self, new_size, based_on=None):
        ev_base = deepcopy(based_on) if based_on else deepcopy(self.vehicles[0])
        ev_base.reset(self.network)
        self.vehicles = {}
        for i in range(new_size):
            ev = deepcopy(ev_base)
            ev.id = i
            self.vehicles[i] = ev
        self.set_vehicles_to_route()

        if self.network:
            self.network.setup_cs_capacities_combinations(len(self))

    def plot_cs_occupation(self, figsize=(16, 5)):
        # Charging stations occupation
        fig = plt.figure(figsize=figsize)
        x = np.arange(0, 24*3600, self.network.sample_time_op)
        plt.step(x, self.network.low_res_occupation_container.T)
        plt.title('CS Occupation')
        plt.xlabel('Event')
        plt.ylabel('Number of EVs')
        plt.legend(tuple(f'CS {i}' for i in self.network.charging_stations))
        return fig


def routes_from_csv_folder(folder_path: str, fleet: Fleet):
    routes = {}
    for ev in fleet.vehicles.values():
        path = f'{folder_path}EV{ev.id}_operation.csv'
        df = pd.read_csv(path, index_col=0)
        Sk, Lk = tuple(df.Sk.values), tuple(df.Lk.values)
        x10, x20, x30 = df.x1_leaving.iloc[0], df.x2_leaving.iloc[0], df.x3_leaving.iloc[0]
        routes[ev.id] = ((Sk, Lk), x10, x20, x30)
    return routes


def from_xml(filepath: Union[Path, str]) -> Union[Fleet, GaussianFleet]:
    element = ET.parse(filepath).getroot()
    if element.tag == 'fleet':
        _fleet = element
    else:
        _fleet = element.find('fleet')
    t = _fleet.get('type')
    cls = globals()[t]
    return cls.from_xml(element)
