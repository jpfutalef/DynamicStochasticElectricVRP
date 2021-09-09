import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from pathlib import Path
import copy

import res.models.Fleet as Fleet
import res.models.Network as Network
from res.tools.IOTools import RouteDict, DepartDict, write_routes, read_routes, write_pretty_xml

from res.optimizer.GATools import OnGA_HyperParameters
from res.optimizer import onGA


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
    departure_time: float = 0.0
    cumulated_consumed_energy: float = 0.0
    min_soc: float = 0.0
    max_soc: float = 0.0

    def xml_element(self) -> [ET.Element]:
        return ET.Element('measurement', attrib={str(i): str(j) for i, j in self.__dict__.items()})

    def update_max_soc(self, soc: float):
        if soc > self.max_soc:
            self.max_soc = soc
        return

    def update_min_soc(self, soc: float):
        if soc < self.min_soc:
            self.min_soc = soc
        return

    def update_from_element(self, etree: ET.Element):
        for key, attrib in etree.attrib.items():
            if attrib == 'True':
                attrib = True
            elif attrib == 'False':
                attrib = False
            elif attrib == 'None':
                attrib = None
            self.__dict__[key] = type(self.__dict__[key])(attrib)


def create_measurements_file(filepath: Path, routes_path: Path, start_earlier_by: float = 0.,
                             based_on: Path = None) -> Tuple[Dict[int, ElectricVehicleMeasurement], ET.Element]:
    routes, departure_info = read_routes(routes_path, read_depart_info=True)
    if based_on:
        collection = read_measurements(based_on)
    else:
        collection = {}
        for id_ev in routes.keys():
            collection[id_ev] = ElectricVehicleMeasurement(id_ev)
            collection[id_ev].min_soc = departure_info[id_ev][1]
            collection[id_ev].max_soc = departure_info[id_ev][1]
    t = min([info[0] for info in departure_info.values()]) - start_earlier_by
    for id_ev in routes.keys():
        route = routes[id_ev]
        info = departure_info[id_ev]
        m = collection[id_ev]

        m.stopped_at_node_from = True
        m.node_from = route[0][0]
        m.node_to = route[0][1]
        m.eta = 0.

        m.time_finishing_service = info[0]
        m.soc_finishing_service = info[1]
        m.payload_finishing_service = info[2]

        m.time = t
        m.departure_time = departure_info[id_ev][0]
        m.soc = info[1]
        m.payload = info[2]

        m.visited_nodes = 1
        m.done = False
        # collection[id_ev] = m

    root = ET.Element('measurements')
    [root.append(m.xml_element()) for m in collection.values()]
    ET.ElementTree(root).write(filepath)
    return collection, root


def read_measurements(filepath: Path) -> Dict[int, ElectricVehicleMeasurement]:
    measurements = {}
    root = ET.parse(filepath).getroot()
    for _measurement in root:
        id_ev = int(_measurement.get('id'))
        measurements[id_ev] = ElectricVehicleMeasurement(id_ev)
        measurements[id_ev].update_from_element(_measurement)
    return measurements


def update_measurements(filepath: Path, measurements: Dict[int, ElectricVehicleMeasurement]):
    root = ET.parse(filepath).getroot()
    for _measurement in root:
        measurements[int(_measurement.get('id'))].update_from_element(_measurement)
    return measurements


def write_measurements(filepath: Path, measurements: Dict[int, ElectricVehicleMeasurement], write_pretty: bool = False):
    root = ET.Element('measurements')
    for measurement in measurements.values():
        root.append(measurement.xml_element())
    if write_pretty:
        write_pretty_xml(filepath, root)
        return
    ET.ElementTree(root).write(filepath)


@dataclass
class Dispatcher:
    network_path: Path
    fleet_path: Path
    measurements_path: Path
    routes_path: Path

    network: Network.Network = None
    fleet: Fleet.Fleet = None

    ga_time: float = 300.  # min
    offset_time: float = 60.  # min
    measurements: Dict[int, ElectricVehicleMeasurement] = None
    routes: RouteDict = None
    depart_info: DepartDict = None
    time: float = None
    onGA_hyper_parameters: OnGA_HyperParameters = None
    exec_times: List = None

    def __post_init__(self):
        self.read_measurements()
        self.update_fleet()
        self.update_network()
        self.update_routes(read_depart_info=True)
        t = min([m.time_finishing_service for m in self.measurements.values()])
        self.init_time = t
        self.time = t
        self.exec_times = []

    def done(self):
        for meas in self.measurements.values():
            if not meas.done:
                return False
        return True

    def update_network(self):
        self.network = Network.from_xml(self.network_path)

    def update_fleet(self):
        self.fleet = Fleet.from_xml(self.fleet_path)

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

    def synchronization(self) -> Tuple[Dict[int, onGA.CriticalPoint], Dict[int, Tuple]]:
        critical_nodes = {}
        routes_with_critical_nodes = {}
        for id_ev, meas in self.measurements.items():
            # Vehicle finished the operation. Pass.
            if meas.done:
                continue

            """
            FIRST - Find next node the EV will visit and iterate from there
            """
            j0 = meas.visited_nodes - 1
            route = self.routes[id_ev]
            (S, L, w1) = route[0][j0:], route[1][j0:], route[2][j0:]

            # Few customers ahead. Pass.
            if sum([1 for i in S if self.network.is_customer(i)]) <= 1:
                continue

            # Get the EV
            ev = self.fleet.vehicles[id_ev]

            # EV is performing an operation at the first node of S
            if meas.stopped_at_node_from:
                departure_time = meas.time_finishing_service + w1[j0]  # TODO check this
                departure_soc = meas.soc_finishing_service
                departure_payload = meas.payload_finishing_service

                if S[0] == 0:
                    # Consider this node as the critical node.
                    j1 = 0
                    Sj1_arrival_time = departure_time
                    Sj1_sos_soc = departure_soc
                    Sj1_sos_payload = departure_payload

                else:
                    # Arrival state at next node
                    j1 = 1
                    Sj1_arrival_time = departure_time + self.network.t(S[0], S[1], departure_time)
                    Sj1_sos_soc = departure_soc - self.network.E(S[0], S[1], departure_payload + ev.weight,
                                                                 departure_time) * 100 / ev.battery_capacity
                    Sj1_sos_payload = departure_payload

            else:
                j1 = 1
                eta = meas.eta

                # Arrival state at next node
                Sj1_arrival_time = meas.time + (1 - eta) * self.network.t(S[0], S[1], meas.time)
                Sj1_sos_soc = meas.soc - 100. * (1 - eta) * self.network.E(S[0], S[1], meas.payload + ev.weight,
                                                                           meas.time) / ev.battery_capacity
                Sj1_sos_payload = meas.payload

            nS = S[j1:]
            nL = L[j1:]
            nw1 = w1[j1:]

            Lj1 = L[j1] if Sj1_sos_soc + L[j1] <= ev.alpha_up else ev.alpha_up - Sj1_sos_soc

            Sj1_low_tw = self.network.nodes[S[j1]].time_window_low
            Sj1_sos_time = Sj1_arrival_time if Sj1_arrival_time > Sj1_low_tw else Sj1_low_tw
            Sj1_service_time = self.network.spent_time(S[j1], Sj1_sos_soc, Lj1)

            # Iterate from the first non-visited node
            ev.set_route(nS, nL, Sj1_sos_time, Sj1_sos_soc, Sj1_sos_payload)
            ev.step(self.network)

            """
            SECOND - Calculate critical points and states
            """
            for k, (sos_time, S0, L0) in enumerate(zip(ev.state_reaching[0, :], nS, nL)):
                if sos_time - meas.time >= self.ga_time + self.offset_time:
                    # Few customers ahead. Pass.
                    if sum([1 for i in nS[k:] if self.network.is_customer(i)]) <= 1:
                        break

                    j_critical = j0 + j1 + k
                    x1_critical = float(ev.state_reaching[0, k])
                    x2_critical = float(ev.state_reaching[1, k])
                    x3_critical = float(ev.state_reaching[2, k])
                    ev.assigned_customers = tuple(i for i in nS if self.network.is_customer(i))
                    critical_nodes[id_ev] = onGA.CriticalPoint(j_critical, S0, L0, x1_critical, x2_critical,
                                                               x3_critical)
                    routes_with_critical_nodes[id_ev] = (nS[k:], nL[k:])
                    ev.current_max_tour_duration = ev.max_tour_duration + meas.departure_time - x1_critical
                    break
        return critical_nodes, routes_with_critical_nodes

    def synchronization_legacy(self) -> Tuple[Dict[int, onGA.CriticalPoint], Dict[int, Tuple]]:
        critical_nodes = {}
        routes_with_critical_nodes = {}
        for id_ev, meas in self.measurements.items():
            # Vehicle finished the operation. Pass.
            if meas.done:
                continue

            """
            FIRST - Find next node the EV will visit and iterate from there
            """
            j0 = meas.visited_nodes - 1
            route = self.routes[id_ev]
            (S, L, w1) = route[0][j0:], route[1][j0:], route[2][j0:]

            # Few customers ahead. Pass.
            if sum([1 for i in S if self.network.is_customer(i)]) <= 1:
                continue

            # Get the EV
            ev = self.fleet.vehicles[id_ev]

            # EV is performing an operation at the first node of S
            if meas.stopped_at_node_from:
                departure_time = meas.time_finishing_service + w1[0]  # TODO check this
                departure_soc = meas.soc_finishing_service
                departure_payload = meas.payload_finishing_service

                if S[0] == 0:
                    # Consider this node as the critical node.
                    j1 = 0
                    Sj1_arrival_time = departure_time
                    Sj1_sos_soc = departure_soc
                    Sj1_sos_payload = departure_payload

                else:
                    # Arrival state at next node
                    j1 = 1
                    Sj1_arrival_time = departure_time + self.network.t(S[0], S[1], departure_time)
                    Sj1_sos_soc = departure_soc - self.network.E(S[0], S[1], departure_payload + ev.weight,
                                                                 departure_time) * 100 / ev.battery_capacity
                    Sj1_sos_payload = departure_payload

            else:
                j1 = 1
                eta = meas.eta

                # Arrival state at next node
                Sj1_arrival_time = meas.time + (1 - eta) * self.network.t(S[0], S[1], meas.time)
                Sj1_sos_soc = meas.soc - 100. * (1 - eta) * self.network.E(S[0], S[1], meas.payload + ev.weight,
                                                                           meas.time) / ev.battery_capacity
                Sj1_sos_payload = meas.payload

            nS = S[j1:]
            nL = L[j1:]
            nw1 = w1[j1:]

            Lj1 = nL[0] if Sj1_sos_soc + nL[0] <= ev.alpha_up else ev.alpha_up - Sj1_sos_soc

            Sj1_low_tw = self.network.nodes[nS[0]].time_window_low
            Sj1_sos_time = Sj1_arrival_time if Sj1_arrival_time > Sj1_low_tw else Sj1_low_tw
            Sj1_service_time = self.network.spent_time(nS[0], Sj1_sos_soc, Lj1)

            x1_0 = Sj1_sos_time + Sj1_service_time + nw1[0]
            x2_0 = Sj1_sos_soc + Lj1
            x3_0 = Sj1_sos_payload - self.network.demand(nS[0])

            # Iterate from the first non-visited node
            reaching_state = np.asarray([Sj1_sos_time, Sj1_sos_soc, Sj1_sos_payload])
            ev.set_route(nS, nL, x1_0, x2_0, x3_0)
            # ev.set_route((nS, nL), x1_0, x2_0, x3_0, reaching_state=reaching_state)
            ev.step(self.network)

            """
            SECOND - Calculate critical points and states
            """
            for k, (sos_time, S0, L0) in enumerate(zip(ev.state_reaching[0, :], nS, nL)):
                if sos_time - meas.time >= self.ga_time + self.offset_time:
                    # Few customers ahead. Pass.
                    if sum([1 for i in nS[k:] if self.network.is_customer(i)]) <= 1:
                        break

                    j_critical = j0 + j1 + k
                    x1_critical = float(ev.state_reaching[0, k])
                    x2_critical = float(ev.state_reaching[1, k])
                    x3_critical = float(ev.state_reaching[2, k])
                    ev.assigned_customers = tuple(i for i in nS if self.network.is_customer(i))
                    critical_nodes[id_ev] = onGA.CriticalPoint(j_critical, S0, L0, x1_critical, x2_critical,
                                                               x3_critical)
                    routes_with_critical_nodes[id_ev] = (nS[k:], nL[k:])
                    ev.current_max_tour_duration = ev.max_tour_duration + meas.departure_time - x1_critical
                    break
        return critical_nodes, routes_with_critical_nodes

    def optimize_online(self, exec_time_filepath):
        critical_points, routes_with_critical_points = self.synchronization()
        if not bool(critical_points):
            return
        self.fleet.set_vehicles_to_route([id_ev for id_ev in critical_points.keys()])
        best_ind = onGA.code(self.fleet, routes_with_critical_points, r=self.onGA_hyper_parameters.r)
        routes, toolbox, report = onGA.onGA(self.fleet, self.onGA_hyper_parameters, critical_points, save_to=None,
                                            best_ind=best_ind)

        # Save execution time
        self.exec_times.append(report.algo_time)
        pd.Series(self.exec_times, name='execution_time (seg)').to_csv(exec_time_filepath)

        for id_ev in self.fleet.vehicles_to_route:
            j_critical = critical_points[id_ev][0]
            S_ahead = routes[id_ev][0]
            L_ahead = routes[id_ev][1]
            # w1_ahead = tuple(self.fleet.vehicles[id_ev].waiting_times0)

            S = self.routes[id_ev][0][:j_critical] + S_ahead
            L = self.routes[id_ev][1][:j_critical] + L_ahead
            # w1 = self.routes[id_ev][2][:j_critical] + w1_ahead
            w1 = (0, ) * len(S)

            self.routes[id_ev] = (tuple(S), tuple(L), tuple(w1))
            if S_ahead[0] == 0:
                self.depart_info[id_ev] = (routes[id_ev][2], routes[id_ev][3], routes[id_ev][4])
        self.write_routes()
