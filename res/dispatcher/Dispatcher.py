import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Tuple, List, Union, Type
import pandas as pd
import numpy as np
from pathlib import Path
import copy

import res.models.Fleet as Fleet
import res.models.Network as Network
import res.models.NonLinearOps as NL
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

    ev_type: Type[Fleet.EV.ElectricVehicle]  = None
    fleet_type: Type[Fleet.Fleet] = None
    edge_type: Type[Network.Edge.DynamicEdge] = None
    network_type: Type[Network.Network] = None

    def __post_init__(self):
        self.read_measurements()
        self.update_fleet()
        self.update_network()
        self.read_routes(read_depart_info=True)
        t = min([m.time_finishing_service for m in self.measurements.values()])
        self.init_time = t
        self.time = t
        self.exec_times = []

    def update(self):
        self.update_network()
        self.update_measurements()
        # self.read_routes()
        self.fleet.set_network(self.network)

    def done(self):
        for meas in self.measurements.values():
            if not meas.done:
                return False
        return True

    def update_network(self):
        self.network = Network.from_xml(self.network_path)

    def update_measurements(self):
        update_measurements(self.measurements_path, self.measurements)

    def update_fleet(self):
        self.fleet = Fleet.from_xml(self.fleet_path)

    def set_measurements(self, measurements: Dict[int, ElectricVehicleMeasurement]):
        self.measurements = measurements

    def read_routes(self, read_depart_info: bool = False):
        self.routes, depart_info = read_routes(self.routes_path, read_depart_info)
        if read_depart_info:
            self.depart_info = depart_info

    def write_routes(self, write_pretty: bool = False):
        write_routes(self.routes_path, self.routes, self.depart_info, write_pretty)

    def set_routes(self, routes: RouteDict, depart_info: DepartDict = None):
        self.routes = routes
        self.depart_info = depart_info if depart_info else self.depart_info

    def read_measurements(self):
        self.measurements = read_measurements(self.measurements_path)

    def write_measurements(self, write_pretty=False):
        write_measurements(self.measurements_path, self.measurements, write_pretty)

    def synchronization(self) -> onGA.CriticalStateDict:
        """
        Applies state synchronization according to stored measurements.
        @return: Dictionary where keys are EV ids, and values the critical state, nodes sequences, and recharging plan
        for that EV, previous to perform online optimization.
        """
        critical_nodes = {}
        for id_ev, meas in self.measurements.items():
            info = self.synchronization_single_ev(id_ev)
            if info is None:
                continue
            elif sum([1 for i in info[1] if self.network.is_customer(i)]) <= 1:
                continue
            else:
                critical_nodes[id_ev] = info

        return critical_nodes

    def synchronization_single_ev(self, id_ev: int) -> Union[onGA.CriticalStateTuple, None]:
        """
        Applies state synchronization for a single EV
        @return: The critical state, nodes sequences, and recharging plan for that EV, previous to perform online
        optimization.
        """
        # Get measurement
        meas = self.measurements[id_ev]

        # CASE: EV already finished operation; do nothing.
        if meas.done:
            return None

        # Obtain current position of the EV
        route = self.routes[id_ev]
        j_curr = meas.visited_nodes - 1
        S_pre, L_pre, w1_pre = route[0][j_curr:], route[1][j_curr:], route[2][j_curr:]

        # CASE: one customer ahead; do nothing.
        if sum([1 for i in S_pre if self.network.is_customer(i)]) <= 1:
            return None

        # Get the EV and the network
        ev = self.fleet.vehicles[id_ev]
        n = self.network

        departure_time = meas.time_finishing_service + w1_pre[0]  # TODO check
        departure_soc = meas.soc_finishing_service
        departure_payload = meas.payload_finishing_service

        if meas.stopped_at_node_from and S_pre[0] == 0:
            # Anchor node is the depot
            j_anchor = 0
            anchor_arrival_time = departure_time
            anchor_arrival_soc = departure_soc
            anchor_arrival_payload = departure_payload

        elif meas.stopped_at_node_from:
            # Anchor node is the next node in the remaining route
            j_anchor = 1
            anchor_arrival_time = departure_time + n.t(S_pre[0], S_pre[1], departure_time)
            anchor_arrival_soc = departure_soc - 100. * n.E(S_pre[0], S_pre[1], departure_payload + ev.weight,
                                                            departure_time) / ev.battery_capacity
            anchor_arrival_payload = departure_payload

        else:
            # Anchor node is the next node the EV will arrive at
            j_anchor = 1
            eta = meas.eta
            anchor_arrival_time = meas.time + (1 - eta) * n.t(S_pre[0], S_pre[1], departure_time)
            anchor_arrival_soc = meas.soc - 100. * (1 - eta) * n.E(S_pre[0], S_pre[1], departure_payload + ev.weight,
                                                                   departure_time) / ev.battery_capacity
            anchor_arrival_payload = meas.payload

        S_anchor = S_pre[j_anchor:]
        L_anchor = L_pre[j_anchor:]
        w1_anchor = w1_pre[j_anchor:]

        anchor_low_tw = self.network.nodes[S_pre[j_anchor]].time_window_low
        anchor_sos_time = NL.saturate(anchor_arrival_time, anchor_low_tw)

        # Iterate from the first non-visited node
        ev.set_route(S_anchor, L_anchor, anchor_sos_time, anchor_arrival_soc, anchor_arrival_payload, w1_anchor[0])
        ev.step(n)

        # Find critical nodes
        for k, (sos_time, Sk, Lk) in enumerate(zip(ev.state_reaching[0, :], S_anchor, L_anchor)):
            if sos_time - meas.time >= self.ga_time + self.offset_time:
                j_critical = j_curr + j_anchor + k
                S_critical = Sk
                L_critical = Lk
                x1_critical = ev.state_reaching[0, k]
                x2_critical = ev.state_reaching[1, k]
                x3_critical = ev.state_reaching[2, k]

                critical_state = onGA.CriticalState(j_critical, S_critical, L_critical, x1_critical, x2_critical,
                                                    x3_critical)

                S_post = S_anchor[k:]
                L_post = L_anchor[k:]

                ev.assigned_customers = tuple(i for i in S_post if n.is_customer(i))
                ev.current_max_tour_duration = ev.max_tour_duration + meas.departure_time - x1_critical

                return critical_state, S_post, L_post
        return None

    def optimize_online(self, exec_time_filepath):
        # Apply synchronization layer
        critical_states = self.synchronization()

        # If no critical nodes/states are found, then do nothing
        if not bool(critical_states):
            return

        # Reset EVs to route according to critical nodes
        self.fleet.set_vehicles_to_route([i for i in critical_states.keys()])

        # Develop candidate individual using previous solution
        best_ind = onGA.code(self.fleet, critical_states, self.onGA_hyper_parameters)

        # Optimize
        routes, toolbox, report = onGA.onGA(self.fleet, self.onGA_hyper_parameters, critical_states, save_to=None,
                                            best_ind=best_ind)

        # Save execution time
        self.exec_times.append(report.algo_time)
        pd.Series(self.exec_times, name='execution_time (seg)').to_csv(exec_time_filepath)

        # Update the routes
        for id_ev in self.fleet.vehicles_to_route:
            critical_state, S_anchor, L_anchor = critical_states[id_ev]
            curr_route = self.routes[id_ev]
            post_S, post_L, post_x10, post_x20, post_x30 = routes[id_ev]

            j_crit = critical_state.j_crit
            # w1_ahead = tuple(self.fleet.vehicles[id_ev].waiting_times0)

            S = curr_route[0][:j_crit] + post_S
            L = curr_route[1][:j_crit] + post_L
            # w1 = self.routes[id_ev][2][:j_critical] + w1_ahead
            w1 = (0,) * len(S)

            self.routes[id_ev] = (tuple(S), tuple(L), tuple(w1))
            if post_S[0] == 0:
                self.depart_info[id_ev] = (post_x10, post_x20, post_x30)

        # Export routes to XML file
        self.write_routes()

    def synchronization_legacy(self) -> Dict[int, Tuple[onGA.CriticalState, Tuple[int, ...], Tuple[float, ...]]]:
        """
        Applies state synchronization according to stored measurements.
        @return: Dictionary where keys are EV ids, and values the critical state, nodes sequences, and recharging plan
        for that EV, previous to perform online optimization.
        """
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
                    critical_nodes[id_ev] = onGA.CriticalNodeInfo(j_critical, S0, L0, x1_critical, x2_critical,
                                                                  x3_critical)
                    routes_with_critical_nodes[id_ev] = (nS[k:], nL[k:])
                    ev.current_max_tour_duration = ev.max_tour_duration + meas.departure_time - x1_critical
                    break
        return critical_nodes, routes_with_critical_nodes

    def synchronization_legacy_old(self) -> Tuple[Dict[int, onGA.CriticalStateTuple], Dict[int, Tuple]]:
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
                    critical_nodes[id_ev] = onGA.CriticalNodeInfo(j_critical, S0, L0, x1_critical, x2_critical,
                                                                  x3_critical)
                    routes_with_critical_nodes[id_ev] = (nS[k:], nL[k:])
                    ev.current_max_tour_duration = ev.max_tour_duration + meas.departure_time - x1_critical
                    break
        return critical_nodes, routes_with_critical_nodes
