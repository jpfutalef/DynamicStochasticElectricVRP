import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, NamedTuple, Union
from res.tools.IOTools import write_pretty_xml
from res.optimizer.GATools import HyperParameters

import numpy as np
from scipy.io import loadmat

import dispatcher.Dispatcher as Dispatcher
import models.Fleet as Fleet
import models.Network as Network


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
                 mat_path: str, sample_time: float, main_folder: str = None, std_factor: Tuple = (1., 1.)):
        if main_folder:
            self.network_path = f'{main_folder}{network_path.split("/")[-1][:-4]}_temp.xml'
            self.fleet_path = f'{main_folder}{fleet_path.split("/")[-1][:-4]}_temp.xml'
        else:
            self.network_path = f'{network_path[:-4]}_temp.xml'
            self.fleet_path = f'{fleet_path[:-4]}_temp.xml'

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

    def disturb_network(self):
        disturb_network(self.data, self.network, std_factor=self.std_factor)
        self.network.write_xml(self.network_path, print_pretty=False)

    def write_measurements(self):
        Dispatcher.write_measurements(self.measurements_path, self.measurements, write_pretty=False)

    def write_routes(self):
        Dispatcher.write_routes(self.routes_path, self.routes, depart_info=None, write_pretty=False)

    def update_routes(self):
        self.routes, _ = Dispatcher.read_routes(self.routes_path, read_depart_info=False)

    def forward_vehicle(self, id_ev: int):
        # Select measurement
        measurement = self.measurements[id_ev]

        # If vehicle finished the operation, do nothing
        if measurement.done:
            return

        # Select variables
        S0, S1 = measurement.node_from, measurement.node_to
        current_time = measurement.time
        ahead_time = current_time + self.sample_time
        j_start = measurement.visited_nodes - 1
        S, L, w1 = self.routes[id_ev][0][j_start:], self.routes[id_ev][1][j_start:], self.routes[id_ev][2][j_start:]
        ev = self.fleet.vehicles[id_ev]

        j0 = S.index(S0)
        j1 = S.index(S1)
        departure_time_from_S0 = measurement.time_finishing_service + w1[j0]

        # Vehicle is stopped at a node
        if measurement.stopped_at_node_from:
            # Vehicle departs
            if ahead_time >= departure_time_from_S0:
                tij = self.network.t(S0, S1, departure_time_from_S0)
                Eij = self.network.e(S0, S1, measurement.payload, ev.weight, departure_time_from_S0, tij)
                eij = 100 * Eij / ev.battery_capacity

                measurement.eta = eta = (ahead_time - departure_time_from_S0) / tij
                measurement.soc = measurement.soc_finishing_service - eta * eij
                measurement.payload = measurement.payload_finishing_service

                measurement.stopped_at_node_from = False

                self.history.update_travelled_time(id_ev, ahead_time - departure_time_from_S0)
                self.history.update_consumed_energy(id_ev, eta * Eij)

                if self.network.time_window_upp(S0) < departure_time_from_S0:
                    v = ConstraintViolation('time_window_upp', self.network.time_window_upp(S0), departure_time_from_S0,
                                            S0, 'leaving')
                    self.history.add_vehicle_constraint_violation(id_ev, v)

                if ev.alpha_up < measurement.soc_finishing_service:
                    v = ConstraintViolation('alpha_up', ev.alpha_up, measurement.soc_finishing_service, S0,
                                            'finishing_service')
                    self.history.add_vehicle_constraint_violation(id_ev, v)

                if ev.alpha_down > measurement.soc_finishing_service:
                    v = ConstraintViolation('alpha_down', ev.alpha_down, measurement.soc_finishing_service, S0,
                                            'finishing_service')
                    self.history.add_vehicle_constraint_violation(id_ev, v)

        # Vehicle is moving across an arc
        else:
            tij = self.network.t(S0, S1, departure_time_from_S0)
            Eij = self.network.e(S0, S1, measurement.payload, ev.weight, departure_time_from_S0, tij)
            eij = 100 * Eij / ev.battery_capacity
            delta = self.sample_time / tij

            # Vehicle reaches next node
            if delta > 1 - measurement.eta:
                remaining_arc_portion = 1 - measurement.eta
                time_reaching = current_time + tij * remaining_arc_portion
                soc_reaching = measurement.soc - eij * remaining_arc_portion
                payload_reaching = measurement.payload

                measurement.visited_nodes += 1

                self.history.update_travelled_time(id_ev, tij * remaining_arc_portion)
                self.history.update_consumed_energy(id_ev, Eij * remaining_arc_portion)

                # The node is the depot: the tour ends
                if S1 == 0:
                    measurement.done = True
                    measurement.node_from = 0
                    measurement.node_to = 0
                    measurement.soc = soc_reaching
                    measurement.payload = payload_reaching
                    measurement.time_finishing_service = time_reaching
                    measurement.soc_finishing_service = soc_reaching
                    measurement.payload_finishing_service = payload_reaching

                # The tour continues
                else:
                    tw_low = self.network.nodes[S1].time_window_low
                    waiting_time = tw_low - time_reaching if tw_low > time_reaching else 0.

                    measurement.node_from = S1
                    measurement.node_to = S[j1 + 1]

                    Lj1 = L[j1] if soc_reaching + L[j1] <= 100 else 100 - soc_reaching

                    service_time = self.network.spent_time(S1, soc_reaching, Lj1)
                    eos_time = time_reaching + waiting_time + service_time
                    eos_soc = soc_reaching + Lj1
                    eos_payload = payload_reaching - self.network.demand(S1)

                    measurement.time_finishing_service = eos_time
                    measurement.soc_finishing_service = eos_soc
                    measurement.payload_finishing_service = eos_payload

                    measurement.stopped_at_node_from = True

                    if self.network.isChargingStation(S1):
                        price = self.network.nodes[S1].price
                        self.history.update_recharging_cost(id_ev, price * Lj1 * ev.battery_capacity)
                        self.history.update_recharging_time(id_ev, service_time)

                if ev.alpha_up < soc_reaching:
                    v = ConstraintViolation('alpha_up', ev.alpha_up, soc_reaching, S1, 'arriving')
                    self.history.add_vehicle_constraint_violation(id_ev, v)

                if ev.alpha_down > soc_reaching:
                    v = ConstraintViolation('alpha_down', ev.alpha_down, soc_reaching, S1, 'arriving')
                    self.history.add_vehicle_constraint_violation(id_ev, v)

            # Vehicle continues moving across the arc
            else:
                measurement.eta += delta
                measurement.soc -= delta * eij

                self.history.update_travelled_time(id_ev, self.sample_time)
                self.history.update_consumed_energy(id_ev, delta * Eij)

        # Increase time
        measurement.time += self.sample_time

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
    simulation_number = 15
    std_factor = (12., 12.)
    soc_policy = (20, 95)

    onGA_hyper_parameters = HyperParameters(num_individuals=80, max_generations=150, CXPB=0.7, MUTPB=0.6,
                                            weights=(0.1 / 2.218, 1. / 0.4364, 1. / 100, 1. / 500, 1. * 0.5),
                                            K1=1000, K2=2000, keep_best=1, tournament_size=3, r=2,
                                            alpha_up=soc_policy[1], algorithm_name='onGA', crossover_repeat=1,
                                            mutation_repeat=1)

    net_path = '../../data/online/instance21/init_files/network.xml'
    fleet_path = '../../data/online/instance21/init_files/fleet.xml'
    routes_path = '../../data/online/instance21/init_files/routes.xml'
    mat_path = '../../data/online/instance21/init_files/21_nodes.mat'

    """
    WITHOUT OPTIMIZATION

    online = False
    stage = 'online' if online else 'offline'

    for i in range(simulation_number):
        print(f'--- Simulation ({stage}) #{i} ---')
        main_folder = f'../../data/online/instance21/{stage}/simulation_{i}/'
        measurements_path = f'../../data/online/instance21/{stage}/simulation_{i}/measurements.xml'
        history_path = f'../../data/online/instance21/{stage}/simulation_{i}/history.xml'

        sim = Simulator(net_path, fleet_path, measurements_path, routes_path, history_path, mat_path, 5., main_folder,
                        std_factor=std_factor)
        dispatcher = Dispatcher.Dispatcher(sim.network_path, sim.fleet_path, sim.measurements_path, sim.routes_path,
                                           onGA_hyper_parameters=onGA_hyper_parameters)

        while not sim.done():
            sim.disturb_network()
            if online:
                dispatcher.update()
                dispatcher.optimize_online()
            sim.forward_fleet()
            sim.save_history()
            
    """

    """ 
    WITH OPTIMIZATION
    """
    online = True
    stage = 'online' if online else 'offline'

    for i in range(simulation_number):
        print(f'--- Simulation ({stage}) #{i} ---')
        main_folder = f'../../data/online/instance21/{stage}/simulation_{i}/'
        measurements_path = f'../../data/online/instance21/{stage}/simulation_{i}/measurements.xml'
        history_path = f'../../data/online/instance21/{stage}/simulation_{i}/history.xml'

        sim = Simulator(net_path, fleet_path, measurements_path, routes_path, history_path, mat_path, 5., main_folder,
                        std_factor=std_factor)
        dispatcher = Dispatcher.Dispatcher(sim.network_path, sim.fleet_path, sim.measurements_path, sim.routes_path,
                                           onGA_hyper_parameters=onGA_hyper_parameters)

        while not sim.done():
            sim.disturb_network()
            if online:
                dispatcher.update()
                dispatcher.optimize_online()
            sim.forward_fleet()
            sim.save_history()
