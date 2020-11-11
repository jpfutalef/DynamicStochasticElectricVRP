from dataclasses import dataclass
from typing import List, Dict, Tuple
import models.Network as net
import models.OnlineFleet as fleet
import xml.etree.ElementTree as ET
import numpy as np
import datetime
import copy
import numpy as np


@dataclass
class ElectricVehicleMeasurement:
    id: int = 0
    node_from: int = 0
    node_to: int = 0
    eta: float = 0.
    time: float = 0.
    soc: float = 0.
    payload: float = 0.
    is_in_node_from: bool = True
    init_operation_time: float = 0.
    arrival_time: float = 0.
    arrival_soc: float = 0.
    arrival_payload: float = 0.
    end_time: float = 0.
    end_soc: float = 0.
    end_payload: float = 0.
    time_since_start: float = 0.
    consumption_since_start: float = 0.
    done: bool = True

    def xml_element(self) -> [ET.Element]:
        return ET.Element('measurement', attrib={str(i): str(j) for i, j in self.__dict__.items()})

    def update(self, etree: ET.Element):
        for key, attrib in etree.attrib.items():
            if key == 'done':
                attrib = True if attrib == 'True' else False
            elif key == 'is_in_node_from':
                attrib = True if attrib == 'True' else False
            self.__dict__[key] = type(self.__dict__[key])(attrib)


def create_collection_file(f: fleet.Fleet, save_to: str):
    root = ET.Element('measurements')
    collection = {}
    for ev in f.vehicles.values():
        m = ElectricVehicleMeasurement(ev.id, ev.route[0][0], ev.route[0][1], 0., 0., ev.x2_0, ev.x3_0, True, ev.x1_0,
                                       ev.x1_0, ev.x2_0, ev.x3_0, ev.x1_0, ev.x2_0, ev.x3_0, 0., 0., False)
        m_element = m.xml_element()
        root.append(m_element)
        collection[ev.id] = m
    ET.ElementTree(root).write(save_to)
    return collection


@dataclass
class Observer:
    network_path: str
    fleet_path: str
    measurements_path: str
    ga_time: float = 5.  # min
    offset_time: float = 1.  # min
    measurements: Dict[int, ElectricVehicleMeasurement] = None
    init_time: float = None
    time: float = None

    def __post_init__(self):
        if self.measurements is None:
            self.set_collection()
        t = np.infty
        for measurement in self.measurements.values():
            t = measurement.end_time if measurement.end_time < t else t
        self.init_time = t
        self.time = t

        for measurement in self.measurements.values():
            measurement.time = t

        self.write_collection()

    def set_collection(self):
        self.measurements = {}
        root = ET.parse(self.measurements_path).getroot()
        for _measurement in root:
            self.measurements[int(_measurement.get('id'))] = ElectricVehicleMeasurement()
            self.measurements[int(_measurement.get('id'))].update(_measurement)

    def read_measurements(self):
        root = ET.parse(self.measurements_path).getroot()
        for _measurement in root:
            self.measurements[int(_measurement.get('id'))].update(_measurement)

    def done(self):
        for meas in self.measurements.values():
            if not meas.done:
                return False
        return True

    def write_collection(self, path=None):
        root = ET.Element('measurements')
        for measurement in self.measurements.values():
            root.append(measurement.xml_element())
        if path:
            ET.ElementTree(root).write(path)
        else:
            ET.ElementTree(root).write(self.measurements_path)

    def observe(self) -> Tuple[
        net.Network, fleet.Fleet, fleet.Fleet, Dict[int, Tuple[Tuple[Tuple, Tuple], float, float, float]], Dict[
            int, Tuple[Tuple[Tuple, Tuple], float, float, float]]]:
        n = net.from_xml(self.network_path, False)
        f_original = fleet.from_xml(self.fleet_path, True, True, False, True)
        f = copy.deepcopy(f_original)
        f.set_network(n)
        self.read_measurements()
        current_routes = {}
        ahead_routes = {}

        for id_ev, meas in self.measurements.items():
            # MAIN CASE - Vehicle finished the operation
            if meas.done:
                f.drop_vehicle(id_ev)
                continue

            if id_ev not in f.vehicles.keys():
                continue

            # MAIN CASE - Vehicle is operating
            ev = f.vehicles[id_ev]

            current_routes[id_ev] = (ev.route, ev.x1_0, ev.x2_0, ev.x3_0)

            # Forecast the future state of vehicles to estimate the critical point
            # SUB CASE - Vehicle is stopped: critical point is considered from the next node in the sequence
            if meas.is_in_node_from:
                (S, L) = ev.route
                k = S.index(meas.node_from)
                x10, x20, x30 = meas.end_time, meas.end_soc, meas.end_payload

                # CASE - There are no more customers ahead
                if sum([1 for i in S[k:] if n.isCustomer(i)]) == 0:
                    f.drop_vehicle(id_ev)

                # CASE - There are customers ahead
                else:
                    ev.set_route((S[k:], L[k:]), x10, x20, x30)
                    ev.step(n)
                    ev.state_reaching = ev.state_reaching[:, 1:]
                    ev.state_leaving = ev.state_leaving[:, 1:]
                    ev.route = (ev.route[0][1:], ev.route[1][1:])

            # SUB SUB CASE - Vehicle is moving. It is necessary to accommodate the reaching point in the matrix after
            # iteration
            else:
                (S, L) = ev.route
                k = S.index(meas.node_to)

                # SUB CASE - There are no more customers ahead
                if sum([1 for i in S[k:] if n.isCustomer(i)]) == 0:
                    f.drop_vehicle(id_ev)

                # CASE - There are customers ahead
                else:
                    t_reach = meas.time + (1 - meas.eta) * n.t(meas.node_from, meas.node_to, meas.time)
                    e_reach = meas.soc - 100. * (1 - meas.eta) * n.e(meas.node_from, meas.node_to, meas.payload,
                                                                     ev.weight,
                                                                     meas.time) / ev.battery_capacity
                    w_reach = meas.payload

                    x10 = t_reach + n.spent_time(meas.node_to, e_reach, L[k]) + ev.waiting_times1[k]
                    x20 = e_reach + L[k]
                    x30 = meas.payload - n.demand(meas.node_to)

                    ev.set_route((S[k:], L[k:]), x10, x20, x30)
                    ev.step(n)
                    ev.state_reaching[:, 0] = np.asarray([t_reach, e_reach, w_reach])
                    ev.route = (ev.route[0][k:], ev.route[1][k:])

            # Calculate critical points
            for k, t in enumerate(ev.state_reaching[0, :]):
                if t - meas.time >= self.ga_time + self.offset_time:
                    Scrit, Lcrit = ev.route[0][k:], ev.route[1][k:]
                    ev.assigned_customers = tuple(i for i in Scrit if f.network.isCustomer(i))
                    x1_arrival = ev.state_reaching[0, k]
                    x2_arrival = ev.state_reaching[1, k]
                    x3_arrival = ev.state_reaching[2, k]

                    ahead_routes[id_ev] = ((Scrit, Lcrit), x1_arrival, x2_arrival, x3_arrival)
                    break

        return n, f, f_original, current_routes, ahead_routes
