import xml.etree.ElementTree as ET
from dataclasses import dataclass
from os import makedirs
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple, Union

import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors
import pandas as pd

import res.dispatcher.Dispatcher as Dispatcher
import res.models.Fleet as Fleet
import res.models.Network as Network
from res.tools.IOTools import write_pretty_xml


def measure(data, i, j, day_points, std_factor=(1., 1.)):
    """
    Reads data object, and returns energy consumption and travel times across arc ij with gaussian noise applied.
    :param data: mat file handler
    :param i: node from id
    :param j: node to id
    :param day_points: number of measurements per day
    :param std_factor: std scaling factor for time and energy consumption (in that order). Default: (1.0, 1.0)
    :return: measurements (travel time, energy consumption) in minutes and kWh
    """
    tt, ec = np.zeros(day_points), np.zeros(day_points)
    for t in range(day_points):
        tt_data = data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['time'][0] / 60
        ec_data = data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['soc'][0] * 24

        tt_average, tt_std = np.mean(tt_data), std_factor[0] * np.std(tt_data)
        ec_average, ec_std, = np.mean(ec_data), std_factor[1] * np.std(ec_data)

        tt[t] = tt_average + np.random.normal(0, tt_std)
        ec[t] = ec_average + np.random.normal(0, ec_std)

    return tt, ec


def disturb_network_mat(data: dict, network: Network.Network, std_factor=(1., 1.)):
    """
    Disturbs the network using a matlab .mat file
    @param data: mat file container
    @param network: network instance
    @param std_factor: std gain
    @return: None. Results are stored in the network instance
    """
    for i in network.nodes.keys():
        for j in network.nodes.keys():
            for t in range(len(network.edges[i][j].energy_consumption)):
                tt_average, tt_std = data[i][j][t]['time'][0], data[i][j][t]['time'][1]
                ec_average, ec_std = data[i][j][t]['ec'][0], data[i][j][t]['ec'][1]

                travel_time = abs(np.random.normal(tt_average, 2.5 + std_factor[0] * tt_std))
                energy_consumption = abs(np.random.normal(ec_average, 0.8 + std_factor[1] * ec_std))

                network.edges[i][j].travel_time[t] = travel_time
                network.edges[i][j].energy_consumption[t] = energy_consumption


def disturb_network_instance(network_source: Network.Network, network: Network.Network, std_factor=1.):
    for i in network_source.nodes.keys():
        for j in network_source.nodes.keys():
            edge_source = network_source.edges[i][j]
            edge_target = network.edges[i][j]
            realization = np.random.normal(loc=edge_source.velocity, scale=std_factor * edge_source.velocity_deviation)
            edge_target.velocity = realization


def saturate(val, min_val=-np.infty, max_val=np.infty):
    if val < min_val:
        val = min_val
    if val > max_val:
        val = max_val
    return val


class ConstraintViolation(NamedTuple):
    type: str = 'generic'
    constraint_value: Union[int, float] = None
    real_value: Union[int, float] = None
    where: Union[int, float] = None
    when: str = None


@dataclass
class EVHistory:
    id: int
    travelled_time: float = 0.
    consumed_energy: float = 0.
    recharging_time: float = 0.
    recharging_cost: float = 0.
    violated_constraints: List[ConstraintViolation] = None

    def __post_init__(self):
        self.violated_constraints = []

    def add_violation(self, violated_constraint: ConstraintViolation):
        self.violated_constraints.append(violated_constraint)

    def xml_element(self) -> ET.Element:
        attrib = {key: str(attr) for key, attr in self.__dict__.items() if key != 'violated_constraints'}
        _element = ET.Element('vehicle', attrib=attrib)
        _violated_constraints = ET.SubElement(_element, 'violated_constraints')
        for violated_constraint in self.violated_constraints:
            attrib = {key: str(attr) for key, attr in violated_constraint._asdict().items()}
            _violated_constraint = ET.SubElement(_violated_constraints, 'violated_constraint', attrib=attrib)
        return _element


@dataclass
class FleetHistory:
    travelled_time: float = 0.
    consumed_energy: float = 0.
    recharging_time: float = 0.
    recharging_cost: float = 0.
    violated_constraints: List[ConstraintViolation] = None
    vehicles_history: Dict[int, EVHistory] = None

    def __post_init__(self):
        self.violated_constraints = []

    def create_from_routes(self, routes: Dispatcher.RouteDict):
        self.vehicles_history = {}
        for id_ev in routes.keys():
            self.vehicles_history[id_ev] = EVHistory(id_ev)
        return self

    def save(self, filepath: Path, write_pretty: bool = False):
        root = ET.Element('history')
        attr = {str(key): str(val) for key, val in self.__dict__.items() if key not in ('violated_constraints, '
                                                                                        'vehicles_history')}
        _fleet_history = ET.SubElement(root, 'fleet', attrib=attr)
        _violated_constraints = ET.SubElement(_fleet_history, 'violated_constraints')
        for violated_constraint in self.violated_constraints:
            attrib = {key: str(attr) for key, attr in violated_constraint._asdict().items()}
            _violated_constraint = ET.SubElement(_violated_constraints, 'violated_constraint', attrib=attrib)

        for ev_history in self.vehicles_history.values():
            root.append(ev_history.xml_element())

        if write_pretty:
            write_pretty_xml(filepath, root)
        else:
            ET.ElementTree(root).write(filepath)

    def update_travelled_time(self, id_ev: int, increment: float):
        self.vehicles_history[id_ev].travelled_time += increment
        self.travelled_time += increment

    def update_recharging_time(self, id_ev: int, increment: float):
        self.vehicles_history[id_ev].recharging_time += increment
        self.recharging_time += increment

    def update_recharging_cost(self, id_ev: int, increment: float):
        self.vehicles_history[id_ev].recharging_cost += increment
        self.recharging_cost += increment

    def update_consumed_energy(self, id_ev: int, increment: float):
        self.vehicles_history[id_ev].consumed_energy += increment
        self.consumed_energy += increment

    def add_vehicle_constraint_violation(self, id_ev: int, violation: ConstraintViolation):
        self.vehicles_history[id_ev].violated_constraints.append(violation)

    def add_fleet_constraint_violation(self, violation: ConstraintViolation):
        self.violated_constraints.append(violation)


class Simulator:
    def __init__(self, network_path: Path, fleet_path: Path, measurements_path: Path, routes_path: Path,
                 history_path: Path, mat_path: Union[Path, None], sample_time: float, operation_folder: Path,
                 std_factor: float = 1.0, create_routes_xml: bool = True, create_measurements_xml: bool = True,
                 create_history_xml: bool = True, eta_model: NearestNeighbors = None, eta_table: np.ndarray = None,
                 previous_day_measurements: Dict[int, Dispatcher.ElectricVehicleMeasurement] = None,
                 new_soc_policy: Tuple[int, int] = None):

        self.main_folder = operation_folder

        self.network_path_original = network_path
        self.network_path = Path(operation_folder, 'network_temp.xml')

        self.fleet_path_original = fleet_path
        self.fleet_path = Path(operation_folder, 'fleet_temp.xml')

        self.routes_path_original = routes_path
        self.routes_path = Path(operation_folder, 'routes_temp.xml')

        self.measurements_path = measurements_path
        self.history_path = history_path
        self.mat_path = mat_path
        self.sample_time = sample_time

        makedirs(operation_folder, exist_ok=True)

        self.network_source = Network.from_xml(self.network_path_original)
        self.network = Network.from_xml(network_path)
        self.fleet = Fleet.from_xml(fleet_path)

        self.routes, self.depart_info = Dispatcher.read_routes(routes_path, read_depart_info=True)
        if new_soc_policy:
            self.fleet.new_soc_policy(new_soc_policy[0], new_soc_policy[1])
            self.depart_info = {id_ev: (i[0], new_soc_policy[1], i[2]) for id_ev, i in self.depart_info.items()}

        self.network.write_xml(self.network_path, print_pretty=False)
        self.fleet.write_xml(self.fleet_path, network_in_file=False, assign_customers=False, with_routes=False,
                             print_pretty=False)
        Dispatcher.write_routes(self.routes_path, self.routes, self.depart_info, write_pretty=False)

        self.data = self.data_from_mat_file()
        self.measurements, _ = self.create_measurements_file(previous_day_measurements)
        self.history = FleetHistory().create_from_routes(self.routes)
        self.save_history()

        self.std_factor = std_factor
        self.day_points = int(24 * 60 * 60 / self.network.edges[0][0].sample_time)
        self.eta_model = eta_model
        self.eta_table = eta_table

    def data_from_mat_file(self) -> Union[Dict, None]:
        if self.mat_path is None:
            return None
        raw_data = loadmat(self.mat_path)
        data = {}
        for i in self.network.nodes.keys():
            data[i] = data_i = {}
            for j in self.network.nodes.keys():
                data_i[j] = data_j = {}
                for t in range(len(self.network.edges[i][j].energy_consumption)):
                    data_j[t] = data_t = {}
                    tt_data = raw_data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['time'][0] / 60
                    ec_data = raw_data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['soc'][0] * 24

                    tt_average, tt_std = np.mean(tt_data), np.std(tt_data)
                    ec_average, ec_std = np.mean(ec_data), np.std(ec_data)

                    data_t['time'] = (tt_average, tt_std)
                    data_t['ec'] = (ec_average, ec_std)
        return data

    def create_measurements_file(self, previous_day_measurements: Dict[int, Dispatcher.ElectricVehicleMeasurement]):
        return Dispatcher.create_measurements_file(self.measurements_path, self.routes_path,
                                                   previous_day_measurements=previous_day_measurements)

    def save_history(self):
        self.history.save(self.history_path, write_pretty=False)

    def disturb_network(self, std_factor: Tuple[float, float] = None):
        if self.data:
            self.disturb_network_from_mat(std_factor)
        else:
            self.disturb_network_from_instance(std_factor)
        self.network.write_xml(self.network_path, print_pretty=False)

    def disturb_network_from_mat(self, std_factor: Tuple[float, float] = None):
        if std_factor is None:
            disturb_network_mat(self.data, self.network, std_factor=self.std_factor)
        else:
            disturb_network_mat(self.data, self.network, std_factor=std_factor)

    def disturb_network_from_instance(self, std_factor: float = None):
        if std_factor is None:
            disturb_network_instance(self.network_source, self.network, std_factor=self.std_factor)
        else:
            disturb_network_instance(self.network_source, self.network, std_factor=std_factor)

    def write_measurements(self):
        Dispatcher.write_measurements(self.measurements_path, self.measurements, write_pretty=False)

    def write_routes(self, path: Path = None):
        depart_info = {}
        routes = {}
        for id_ev, m in self.measurements.items():
            m = self.measurements[id_ev]
            depart_info[id_ev] = (m.departure_time, self.depart_info[id_ev][1], self.depart_info[id_ev][2])
            route_no_departure_wt = (0.,) + self.routes[id_ev][2][1:]
            routes[id_ev] = (self.routes[id_ev][0], self.routes[id_ev][1], route_no_departure_wt)

        if path:
            Dispatcher.write_routes(path, routes, depart_info=depart_info, write_pretty=False)
        else:
            Dispatcher.write_routes(self.routes_path, routes, depart_info=depart_info, write_pretty=False)

    def update_routes(self, read_depart_info=False):
        if read_depart_info:
            self.routes, self.depart_info = Dispatcher.read_routes(self.routes_path, read_depart_info)
        else:
            self.routes, _ = Dispatcher.read_routes(self.routes_path, read_depart_info)

    def forward_vehicle(self, id_ev: int, step_time: float = None):
        # Select measurement
        measurement = self.measurements[id_ev]

        # If vehicle finished the operation, do nothing
        if measurement.done:
            return

        # If forward time not passed, set it and iterate from there
        if step_time is None:
            step_time = self.sample_time

        # Select variables
        ev = self.fleet.vehicles[id_ev]

        current_time = measurement.time
        ahead_time = current_time + step_time
        k_start = measurement.visited_nodes - 1
        S, L, w1 = self.routes[id_ev][0][k_start:], self.routes[id_ev][1][k_start:], self.routes[id_ev][2][k_start:]

        S0, S1 = measurement.node_from, measurement.node_to
        j0 = 0
        j1 = 1

        # Get departure info from S0
        departure_time_from_S0 = measurement.time_finishing_service + w1[j0]
        departure_soc_from_S0 = measurement.soc_finishing_service
        departure_payload_from_S0 = measurement.payload_finishing_service

        # Vehicle is stopped at a node and then it departs towards the next stop
        if measurement.stopped_at_node_from and ahead_time >= departure_time_from_S0:
            measurement.time = departure_time_from_S0
            measurement.soc = departure_soc_from_S0
            measurement.payload = departure_payload_from_S0

            measurement.eta = 0.
            measurement.stopped_at_node_from = False

            measurement.max_soc = measurement.soc if measurement.soc > measurement.max_soc else measurement.max_soc

            # Record the moment the EV departs from the depot
            if not S0:
                measurement.departure_time = departure_time_from_S0

            if self.network.time_window_upp(S0) < departure_time_from_S0:
                v = ConstraintViolation('time_window_upp', self.network.time_window_upp(S0), departure_time_from_S0,
                                        S0, 'leaving')
                self.history.add_vehicle_constraint_violation(id_ev, v)

            if ev.alpha_up < measurement.soc_finishing_service:
                v = ConstraintViolation('alpha_up', ev.alpha_up, departure_soc_from_S0, S0, 'finishing_service')
                self.history.add_vehicle_constraint_violation(id_ev, v)

            if ev.alpha_down > measurement.soc_finishing_service:
                v = ConstraintViolation('alpha_down', ev.alpha_down, departure_soc_from_S0, S0, 'finishing_service')
                self.history.add_vehicle_constraint_violation(id_ev, v)

            self.forward_vehicle(id_ev, ahead_time - departure_time_from_S0)

        # Vehicle is stopped at a node and then it remains there
        elif measurement.stopped_at_node_from and ahead_time < departure_time_from_S0:
            measurement.time += step_time

        # Vehicle is traversing an an arc
        else:
            tij = self.network.t(S0, S1, measurement.time)
            Eij = self.network.E(S0, S1, measurement.payload + ev.weight, measurement.time)
            eij = 100 * Eij / ev.battery_capacity
            delta = step_time / tij if tij else 1.0

            # Vehicle reaches next node
            if delta > 1 - measurement.eta:
                remaining_arc_portion = 1 - measurement.eta
                arrival_time = current_time + tij * remaining_arc_portion
                arrival_soc = measurement.soc - eij * remaining_arc_portion
                arrival_payload = measurement.payload

                measurement.visited_nodes += 1
                measurement.stopped_at_node_from = True

                travel_time_portion = tij * remaining_arc_portion
                energy_consumption_portion = Eij * remaining_arc_portion
                self.history.update_travelled_time(id_ev, travel_time_portion)
                self.history.update_consumed_energy(id_ev, energy_consumption_portion)

                if ev.alpha_up < arrival_soc:
                    v = ConstraintViolation('alpha_up', ev.alpha_up, arrival_soc, S1, 'arriving')
                    self.history.add_vehicle_constraint_violation(id_ev, v)

                if ev.alpha_down > arrival_soc:
                    v = ConstraintViolation('alpha_down', ev.alpha_down, arrival_soc, S1, 'arriving')
                    self.history.add_vehicle_constraint_violation(id_ev, v)

                # The node is the depot: the tour ends
                if not S1:
                    measurement.done = True
                    measurement.node_from = 0
                    measurement.node_to = 0
                    measurement.soc = arrival_soc
                    measurement.payload = arrival_payload
                    measurement.time_finishing_service = arrival_time
                    measurement.soc_finishing_service = arrival_soc
                    measurement.payload_finishing_service = arrival_payload
                    measurement.time = arrival_time
                    measurement.min_soc = measurement.soc if measurement.soc < measurement.min_soc else measurement.min_soc
                    if arrival_time - measurement.departure_time > ev.max_tour_duration:
                        v = ConstraintViolation('max_tour_time', ev.max_tour_duration,
                                                arrival_time - measurement.departure_time, 0, 'arriving')
                        self.history.add_vehicle_constraint_violation(id_ev, v)

                # The tour continues
                else:
                    tw_low = self.network.nodes[S1].time_window_low
                    waiting_time = tw_low - arrival_time if tw_low > arrival_time else 0.

                    measurement.node_from = S1
                    measurement.node_to = S[j1 + 1]

                    Lj1 = L[j1] if arrival_soc + L[j1] <= ev.alpha_up else ev.alpha_up - arrival_soc

                    service_time = self.network.spent_time(S1, arrival_soc, Lj1)
                    eos_time = arrival_time + waiting_time + service_time
                    eos_soc = arrival_soc + Lj1
                    eos_payload = arrival_payload - self.network.demand(S1)

                    measurement.time_finishing_service = eos_time
                    measurement.soc_finishing_service = eos_soc
                    measurement.payload_finishing_service = eos_payload

                    measurement.time = arrival_time
                    measurement.soc = arrival_soc
                    measurement.payload = arrival_payload

                    measurement.min_soc = measurement.soc if measurement.soc < measurement.min_soc else measurement.min_soc

                    if self.network.is_charging_station(S1):
                        price = self.network.nodes[S1].price
                        self.history.update_recharging_cost(id_ev, price * Lj1 * ev.battery_capacity)
                        self.history.update_recharging_time(id_ev, service_time)

                    self.forward_vehicle(id_ev, ahead_time - arrival_time)

            # Vehicle continues traversing the arc
            else:
                measurement.time += step_time
                measurement.soc -= delta * eij
                measurement.eta += delta

                self.history.update_travelled_time(id_ev, step_time)
                self.history.update_consumed_energy(id_ev, delta * Eij)

                energy_consumption_portion = delta * Eij

            # Update cumulated energy consumption
            measurement.cumulated_consumed_energy += energy_consumption_portion

            # Degrade if an energy consumption equal to the nominal capacity is reached
            # if self.eta_model and measurement.cumulated_consumed_energy > ev.battery_capacity_nominal:
            #     eta_val = ev.degrade_battery(self.eta_table, self.eta_model, measurement.min_soc, measurement.max_soc)
            #
            #     # Save this data
            #     data = {'time': [measurement.time], 'eta': [eta_val], 'min soc': [measurement.min_soc],
            #             'max soc': [measurement.max_soc]}
            #     path = Path(self.main_folder, 'degradation_event.csv')
            #     with_columns = False if path.is_file() else True
            #     with open(path, 'a') as file:
            #         pd.DataFrame(data).to_csv(file, index=False, header=with_columns)
            #
            #     # Reset value
            #     measurement.cumulated_consumed_energy -= ev.battery_capacity_nominal
            #     measurement.min_soc = measurement.soc
            #     measurement.max_soc = measurement.soc

    def forward_fleet(self):
        self.update_routes()
        for id_ev in self.measurements.keys():
            self.forward_vehicle(id_ev)
        self.write_measurements()

    def done(self):
        for m in self.measurements.values():
            if not m.done:
                return False
        return True
