import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, NamedTuple, Union
from res.tools.IOTools import write_pretty_xml
from res.optimizer.GATools import HyperParameters

import numpy as np
from scipy.io import loadmat
from datetime import datetime
from os import makedirs

import res.dispatcher.Dispatcher as Dispatcher
import res.models.Fleet as Fleet
import res.models.Network as Network


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


def disturb_network(data: dict, network: Network.Network, std_factor=(1., 1.)):
    for i in network.nodes.keys():
        for j in network.nodes.keys():
            for t in range(len(network.edges[i][j].energy_consumption)):
                tt_average, tt_std = data[i][j][t]['time'][0], data[i][j][t]['time'][1]
                ec_average, ec_std = data[i][j][t]['ec'][0], data[i][j][t]['ec'][1]

                network.edges[i][j].travel_time[t] = tt_average + np.random.normal(0, std_factor[0] * tt_std)
                network.edges[i][j].energy_consumption[t] = ec_average + np.random.normal(0, std_factor[1] * ec_std)


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

    def save(self, filepath: str, write_pretty: bool = False):
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
    def __init__(self, network_path: str, fleet_path: str, measurements_path: str, routes_path: str, history_path: str,
                 mat_path: str, sample_time: float, main_folder: str = None, std_factor: Tuple = (1., 1.),
                 create_routes_xml: bool=True, create_measurements_xml: bool = True, create_history_xml: bool=True):
        if main_folder:
            self.network_path = f'{main_folder}{network_path.split("/")[-1][:-4]}_temp.xml'
            self.fleet_path = f'{main_folder}{fleet_path.split("/")[-1][:-4]}_temp.xml'
        else:
            self.network_path = net_path
            self.fleet_path = fleet_path

        self.measurements_path = measurements_path
        self.history_path = history_path
        self.mat_path = mat_path
        self.sample_time = sample_time

        self.network = Network.from_xml(network_path, instance=False)
        self.fleet = Fleet.from_xml(fleet_path, assign_customers=False, with_routes=False, instance=False)

        self.network.write_xml(self.network_path, print_pretty=False)
        self.fleet.write_xml(self.fleet_path, network_in_file=False, assign_customers=False, with_routes=False,
                             print_pretty=False)

        self.day_points = int(1440 / self.network.edges[0][0].sample_time)

        self.data = self.data_from_mat_file()

        if main_folder:
            self.routes_path = f'{main_folder}{routes_path.split("/")[-1]}'
        else:
            self.routes_path = routes_path

        self.routes, self.depart_info = Dispatcher.read_routes(routes_path, read_depart_info=True)
        Dispatcher.write_routes(self.routes_path, self.routes, self.depart_info, write_pretty=False)

        self.measurements, _ = self.create_measurements_file()
        self.history = FleetHistory().create_from_routes(self.routes)
        self.save_history()

        self.std_factor = std_factor

    def data_from_mat_file(self) -> Dict:
        raw_data = loadmat(self.mat_path)
        data = {}
        for i in self.network.nodes.keys():
            data[i] = data_i = {}
            for j in self.network.nodes.keys():
                data_i[j] = data_j = {}
                for t in range(len(self.network.edges[i][j].energy_consumption)):
                    data_j[t] = data_t = {}
                    tt_data = 1.8 * raw_data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['time'][0] / 60
                    ec_data = 1.8 * raw_data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['soc'][0] * 24

                    tt_average, tt_std = np.mean(tt_data), np.std(tt_data)
                    ec_average, ec_std = np.mean(ec_data), np.std(ec_data)

                    data_t['time'] = (tt_average, tt_std)
                    data_t['ec'] = (ec_average, ec_std)
        return data

    def create_measurements_file(self):
        return Dispatcher.create_measurements_file(self.measurements_path, self.routes_path)

    def save_history(self):
        self.history.save(self.history_path, write_pretty=False)

    def disturb_network(self, std_factor: Tuple[float, float] = None):
        if std_factor is None:
            disturb_network(self.data, self.network, std_factor=self.std_factor)
        else:
            disturb_network(self.data, self.network, std_factor=std_factor)
        self.network.write_xml(self.network_path, print_pretty=False)

    def write_measurements(self):
        Dispatcher.write_measurements(self.measurements_path, self.measurements, write_pretty=False)

    def write_routes(self):
        Dispatcher.write_routes(self.routes_path, self.routes, depart_info=None, write_pretty=False)

    def update_routes(self):
        self.routes, self.depart_info = Dispatcher.read_routes(self.routes_path, read_depart_info=True)

        for id_ev, m in self.measurements.items():
            if m.stopped_at_node_from and m.node_from == 0:
                m.time_finishing_service = self.depart_info[id_ev][0]

    def forward_vehicle(self, id_ev: int, forward_time: float = None):
        # Select measurement
        measurement = self.measurements[id_ev]

        # If vehicle finished the operation, do nothing
        if measurement.done:
            return

        # If forward time not passed, set it and iterate from there
        if forward_time is None:
            forward_time = self.sample_time

        # Select variables
        ev = self.fleet.vehicles[id_ev]

        current_time = measurement.time
        ahead_time = current_time + forward_time
        j_start = measurement.visited_nodes - 1
        S, L, w1 = self.routes[id_ev][0][j_start:], self.routes[id_ev][1][j_start:], self.routes[id_ev][2][j_start:]

        S0, S1 = measurement.node_from, measurement.node_to
        j0 = S.index(S0)
        j1 = S.index(S1)

        # Get departure info from S0
        departure_time_from_S0 = measurement.time_finishing_service + w1[j0]
        departure_soc_from_S0 = measurement.soc_finishing_service
        departure_payload_from_S0 = measurement.payload_finishing_service

        # Vehicle is stopped at a node
        if measurement.stopped_at_node_from:
            # Vehicle departs
            if ahead_time >= departure_time_from_S0:
                measurement.time = departure_time_from_S0
                measurement.soc = departure_soc_from_S0
                measurement.payload = departure_payload_from_S0

                measurement.eta = 0.
                measurement.stopped_at_node_from = False

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

            # Vehicles stays at the node
            else:
                measurement.time += forward_time

        # Vehicle is traversing an an arc
        else:
            tij = self.network.t(S0, S1, departure_time_from_S0)
            Eij = self.network.e(S0, S1, measurement.payload, ev.weight, departure_time_from_S0, tij)
            eij = 100 * Eij / ev.battery_capacity
            delta = np.divide(forward_time, tij)

            # Vehicle reaches next node
            if delta > 1 - measurement.eta:
                remaining_arc_portion = 1 - measurement.eta
                arrival_time = current_time + tij * remaining_arc_portion
                arrival_soc = measurement.soc - eij * remaining_arc_portion
                arrival_payload = measurement.payload

                measurement.visited_nodes += 1
                measurement.stopped_at_node_from = True

                self.history.update_travelled_time(id_ev, tij * remaining_arc_portion)
                self.history.update_consumed_energy(id_ev, Eij * remaining_arc_portion)

                if ev.alpha_up < arrival_soc:
                    v = ConstraintViolation('alpha_up', ev.alpha_up, arrival_soc, S1, 'arriving')
                    self.history.add_vehicle_constraint_violation(id_ev, v)

                if ev.alpha_down > arrival_soc:
                    v = ConstraintViolation('alpha_down', ev.alpha_down, arrival_soc, S1, 'arriving')
                    self.history.add_vehicle_constraint_violation(id_ev, v)

                # The node is the depot: the tour ends
                if S1 == 0:
                    measurement.done = True
                    measurement.node_from = 0
                    measurement.node_to = 0
                    measurement.soc = arrival_soc
                    measurement.payload = arrival_payload
                    measurement.time_finishing_service = arrival_time
                    measurement.soc_finishing_service = arrival_soc
                    measurement.payload_finishing_service = arrival_payload
                    if ev.max_tour_duration < arrival_time - self.depart_info[id_ev][0]:
                        v = ConstraintViolation('max_tour_time', ev.max_tour_duration, arrival_time, 0, 'arriving')
                        self.history.add_vehicle_constraint_violation(id_ev, v)

                # The tour continues
                else:
                    tw_low = self.network.nodes[S1].time_window_low
                    waiting_time = tw_low - arrival_time if tw_low > arrival_time else 0.

                    measurement.node_from = S1
                    measurement.node_to = S[j1 + 1]

                    # Lj1 = L[j1] if arrival_soc + L[j1] <= ev.alpha_up else ev.alpha_up - arrival_soc
                    Lj1 = L[j1] if arrival_soc + L[j1] <= 100. else 100. - arrival_soc

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

                    if self.network.isChargingStation(S1):
                        price = self.network.nodes[S1].price
                        self.history.update_recharging_cost(id_ev, price * Lj1 * ev.battery_capacity)
                        self.history.update_recharging_time(id_ev, service_time)

                    self.forward_vehicle(id_ev, ahead_time - arrival_time)

            # Vehicle continues continues traversing the arc
            else:
                measurement.time += forward_time
                measurement.soc -= delta * eij
                measurement.eta += delta

                self.history.update_travelled_time(id_ev, self.sample_time)
                self.history.update_consumed_energy(id_ev, delta * Eij)

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


if __name__ == '__main__':
    main_folder = '../../data/online/instance21/online_withoutTimeWindows/simulation_3/'
    net_path = f'{main_folder}network_temp.xml'
    fleet_path = f'{main_folder}fleet_temp.xml'
    routes_path = f'{main_folder}routes.xml'
    mat_path = f'../../data/online/instance21/init_files/21_nodes.mat'
    measurements_path = f'{main_folder}/measurements.xml'
    history_path = f'{main_folder}/history.xml'
    exec_time_path = f'{main_folder}/exec_time.csv'

    std_factor = (10., 10.)
    soc_policy = (20, 95)
    onGA_hyper_parameters = HyperParameters(num_individuals=80, max_generations=160, CXPB=0.9, MUTPB=0.6,
                                            weights=(0.1 / 2.218, 1. / 0.4364, 1. / 100, 1. / 500, 1.),
                                            K1=100000, K2=200000, keep_best=1, tournament_size=3, r=2,
                                            alpha_up=soc_policy[1], algorithm_name='onGA', crossover_repeat=1,
                                            mutation_repeat=1)

    dispatcher = Dispatcher.Dispatcher(net_path, fleet_path, measurements_path, routes_path,
                                       onGA_hyper_parameters=onGA_hyper_parameters)

    dispatcher.update()
    cp_info = dispatcher.synchronization()
    routes_from_critical_points = {}

    for id_ev in cp_info.keys():
        r = dispatcher.routes[id_ev]
        j_critical = cp_info[id_ev][0]
        x1_0 = cp_info[id_ev][1]
        x2_0 = cp_info[id_ev][2]
        x3_0 = cp_info[id_ev][3]
        ev = dispatcher.fleet.vehicles[id_ev]
        ev.current_max_tour_duration = ev.max_tour_duration + dispatcher.depart_info[id_ev][0] - x1_0

        routes_from_critical_points[id_ev] = ((r[0][j_critical:], r[1][j_critical:]), x1_0, x2_0, x3_0)

    dispatcher.fleet.set_vehicles_to_route([id_ev for id_ev in routes_from_critical_points.keys()])