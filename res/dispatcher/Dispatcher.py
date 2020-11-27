import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

import res.models.Fleet as Fleet
import res.models.Network as Network
from res.tools.IOTools import RouteDict, DepartDict, write_routes, read_routes, write_pretty_xml

from res.optimizer import onGA
from res.optimizer.GATools import HyperParameters


@dataclass
class ElectricVehicleMeasurement:
    id: int
    stopped_at_node_from: bool = False
    node_from: int = 0
    node_to: int = 0
    eta: float = 0.0

    time_finishing_service: float = 0.0
    soc_finishing_service: float = 0.0
    payload_finishing_service: float = 0.0

    time: float = 0.0
    soc: float = 0.0
    payload: float = 0.0

    visited_nodes: int = 0

    done: bool = False

    def xml_element(self) -> [ET.Element]:
        return ET.Element('measurement', attrib={str(i): str(j) for i, j in self.__dict__.items()})

    def update_from_element(self, etree: ET.Element):
        for key, attrib in etree.attrib.items():
            if attrib == 'True':
                attrib = True
            elif attrib == 'False':
                attrib = False
            elif attrib == 'None':
                attrib = None
            self.__dict__[key] = type(self.__dict__[key])(attrib)


def create_measurements_file(filepath: str, routes_path:str) -> Tuple[Dict[int, ElectricVehicleMeasurement],
                                                                      ET.Element]:
    routes, departure_info = read_routes(routes_path, read_depart_info=True)
    collection = {}
    root = ET.Element('measurements')
    t = min([info[0] for info in departure_info.values()])
    for ev_id, route, info in zip(routes.keys(), routes.values(), departure_info.values()):
        m = ElectricVehicleMeasurement(ev_id)
        m.stopped_at_node_from = True
        m.node_from = route[0][0]
        m.node_to = route[0][1]
        m.eta = 0.

        m.time_finishing_service = info[0]
        m.soc_finishing_service = info[1]
        m.payload_finishing_service = info[2]

        m.time = t
        m.soc = info[1]
        m.payload = info[2]

        m.visited_nodes = 1

        m.done = False

        root.append(m.xml_element())
        collection[ev_id] = m
    ET.ElementTree(root).write(filepath)
    return collection, root


def read_measurements(filepath: str) -> Dict[int, ElectricVehicleMeasurement]:
    measurements = {}
    root = ET.parse(filepath).getroot()
    for _measurement in root:
        id_ev = int(_measurement.get('id'))
        measurements[id_ev] = ElectricVehicleMeasurement(id_ev)
        measurements[id_ev].update_from_element(_measurement)
    return measurements


def update_measurements(filepath: str, measurements: Dict[int, ElectricVehicleMeasurement]):
    root = ET.parse(filepath).getroot()
    for _measurement in root:
        measurements[int(_measurement.get('id'))].update_from_element(_measurement)
    return measurements


def write_measurements(filepath: str, measurements: Dict[int, ElectricVehicleMeasurement], write_pretty: bool = False):
    root = ET.Element('measurements')
    for measurement in measurements.values():
        root.append(measurement.xml_element())
    if write_pretty:
        write_pretty_xml(filepath, root)
        return
    ET.ElementTree(root).write(filepath)


@dataclass
class Dispatcher:
    network_path: str
    fleet_path: str
    measurements_path: str
    routes_path: str

    network: Network.Network = None
    fleet: Fleet.Fleet = None

    ga_time: float = 5.  # min
    offset_time: float = 1.  # min
    measurements: Dict[int, ElectricVehicleMeasurement] = None
    routes: RouteDict = None
    depart_info: DepartDict = None
    time: float = None

    onGA_hyper_parameters: HyperParameters = None

    def __post_init__(self):
        self.read_measurements()
        self.update_fleet()
        self.update_network()
        self.update_routes()
        t = min([m.time_finishing_service for m in self.measurements.values()])
        self.init_time = t
        self.time = t

    def done(self):
        for meas in self.measurements.values():
            if not meas.done:
                return False
        return True

    def update_network(self):
        self.network = Network.from_xml(self.network_path, instance=False)

    def update_fleet(self):
        self.fleet = Fleet.from_xml(self.fleet_path, assign_customers=False, with_routes=False, instance=False,
                                    from_online=False)

    def set_routes(self, routes: RouteDict, depart_info: DepartDict = None):
        self.routes = routes
        self.depart_info = depart_info if depart_info else self.depart_info

    def update_routes(self, read_depart_info: bool = False):
        if read_depart_info:
            self.routes, self.depart_info = read_routes(self.routes_path, read_depart_info)
            return
        self.routes, _ = read_routes(self.routes_path)

    def write_routes(self, write_pretty: bool = False):
        write_routes(self.routes_path, self.routes, self.depart_info, write_pretty)

    def set_measurements(self, measurements: Dict[int, ElectricVehicleMeasurement]):
        self.measurements = measurements

    def update_measurements(self):
        update_measurements(self.measurements_path, self.measurements)

    def read_measurements(self):
        self.measurements = read_measurements(self.measurements_path)

    def write_measurements(self, write_pretty=False):
        write_measurements(self.measurements_path, self.measurements, write_pretty)

    def update(self):
        self.update_network()
        self.update_measurements()
        self.update_routes()
        self.fleet.set_network(self.network)

    def synchronization(self) -> Dict[int, Tuple[int, float, float, float]]:
        """
        Applies synchronization procedure to routes, based on the measurements.
        :return: Dictionary containing position of critical node and critical state (j, x1_0, x2_0, x3_0)
        """
        critical_nodes_info = {}
        for id_ev, meas in self.measurements.items():
            # Vehicle finished the operation
            if meas.done:
                continue

            """
            FIRST - Find next node the EV will visit and iterate from there
            """
            j_start = meas.visited_nodes - 1
            (S, L, w1) = self.routes[id_ev][0][j_start:], self.routes[id_ev][1][j_start:], self.routes[id_ev][2][j_start:]
            j_next = S.index(meas.node_to)
            S0, S1 = meas.node_from, meas.node_to

            # There are no more customers ahead
            if sum([1 for i in S[j_next:] if self.network.isCustomer(i)]) <= 1:
                continue

            elif meas.stopped_at_node_from:
                ev = self.fleet.vehicles[id_ev]
                w1_current = w1[j_next - 1]

                departure_time = meas.time_finishing_service + w1_current
                departure_soc = meas.soc_finishing_service
                departure_payload = meas.payload_finishing_service

                # Arrival state at next node
                arrival_time_S1 = departure_time + self.network.t(S0, S1, departure_time)
                arrival_soc_S1 = departure_soc - self.network.e(S0, S1, departure_payload,
                                                             ev.weight, departure_time)*100/ev.battery_capacity
                arrival_payload_S1 = departure_payload

            else:
                ev = self.fleet.vehicles[id_ev]

                eta = meas.eta
                S0, S1 = meas.node_from, meas.node_to

                # Arrival state at next node
                arrival_time_S1 = meas.time + (1 - eta) * self.network.t(S0, S1, meas.time)
                arrival_soc_S1 = meas.soc - 100. * (1 - eta) * self.network.e(S0, S1, meas.payload, ev.weight,
                                                                      meas.time) / ev.battery_capacity
                arrival_payload_S1 = meas.payload

            S_ahead = S[j_next:]
            L_ahead = L[j_next:]
            w1_ahead = w1[j_next:]

            next_node_low_tw = self.network.nodes[S1].time_window_low
            time_at_start_of_service = arrival_time_S1 if arrival_time_S1 > next_node_low_tw else next_node_low_tw
            t_service = self.network.spent_time(S1, arrival_soc_S1, L_ahead[0])

            x1_0 = time_at_start_of_service + t_service + w1_ahead[0]
            x2_0 = arrival_soc_S1 + L_ahead[0]
            x3_0 = arrival_payload_S1 - self.network.demand(S1)

            # Iterate from the first non-visited node
            reaching_state = np.asarray([time_at_start_of_service, arrival_soc_S1, arrival_payload_S1])
            ev.set_route((S_ahead, L_ahead), x1_0, x2_0, x3_0, reaching_state=reaching_state)
            ev.step(self.network)

            #ev.state_reaching[:, 0] = np.asarray([time_at_start_of_service, arrival_soc_S1, arrival_payload_S1])

            """
            SECOND - Calculate critical points and states
            """
            for k, time_at_start_of_service in enumerate(ev.state_reaching[0, :]):
                if time_at_start_of_service - meas.time >= self.ga_time + self.offset_time:
                    j_critical = j_next + k + j_start
                    x1_critical = float(ev.state_reaching[0, k])
                    x2_critical = float(ev.state_reaching[1, k])
                    x3_critical = float(ev.state_reaching[2, k])
                    ev.assigned_customers = tuple(i for i in S_ahead if self.network.isCustomer(i))

                    critical_nodes_info[id_ev] = (j_critical, x1_critical, x2_critical, x3_critical)
                    break

        return critical_nodes_info

    def optimize_online(self):
        cp_info = self.synchronization()
        if not bool(cp_info):
            return
        routes_from_critical_points = {}

        for id_ev in cp_info.keys():
            r = self.routes[id_ev]
            j_critical = cp_info[id_ev][0]
            x1_0 = cp_info[id_ev][1]
            x2_0 = cp_info[id_ev][2]
            x3_0 = cp_info[id_ev][3]

            routes_from_critical_points[id_ev] = ((r[0][j_critical:], r[1][j_critical:]), x1_0, x2_0, x3_0)

        self.fleet.set_vehicles_to_route([id_ev for id_ev in routes_from_critical_points.keys()])
        best_ind, critical_points = onGA.code(self.fleet, routes_from_critical_points, allowed_charging_operations=2)
        routes, opt_data, toolbox = onGA.onGA(self.fleet, self.onGA_hyper_parameters, critical_points, save_to=None,
                                              best_ind=best_ind)
        for id_ev in self.fleet.vehicles_to_route:
            j_critical = cp_info[id_ev][0]
            S_ahead = routes[id_ev][0][0]
            L_ahead = routes[id_ev][0][1]
            w1_ahead = tuple(self.fleet.vehicles[id_ev].waiting_times0)

            S = self.routes[id_ev][0][:j_critical] + S_ahead
            L = self.routes[id_ev][1][:j_critical] + L_ahead
            w1 = self.routes[id_ev][2][:j_critical] + w1_ahead

            self.routes[id_ev] = (S, L, w1)

        self.write_routes()



if __name__ == '__main__':
    network_path = "../../data/online/instance21/init_files/network.xml"
    fleet_path = "../../data/online/instance21/init_files/fleet.xml"
    measurements_path = "../../data/online/instance21/init_files/measurements.xml"
    routes_path = "../../data/online/instance21/init_files/routes.xml"

    d = Dispatcher(network_path, fleet_path, measurements_path, routes_path)

    cp = d.synchronization()

