from dataclasses import dataclass
from typing import List, Dict, Tuple
import models.Network as net
import models.OnlineFleet as fleet
import xml.etree.ElementTree as ET
import numpy as np
import datetime


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
    end_service: float = 0.
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
    for ev in f.vehicles.values():
        m = ElectricVehicleMeasurement(ev.id, ev.route[0][0], ev.route[0][1], 0., 0., ev.x2_0, ev.x3_0, True,
                                       ev.x1_0, ev.x2_0, ev.x3_0, 0., 0., False)
        m_element = m.xml_element()
        root.append(m_element)
    ET.ElementTree(root).write(save_to)


@dataclass
class Observer:
    network_path: str
    fleet_path: str
    measure_path: str
    report_folder: str
    ga_time: float
    offset_time: float
    num_of_last_nodes: int
    collection: Dict[int, ElectricVehicleMeasurement] = None
    init_time: float = None
    time: float = None

    def __post_init__(self):
        if self.collection is None:
            self.set_collection()
        t = 1000000000.
        for meas in self.collection.values():
            t = meas.end_service if meas.end_service < t else t
        self.init_time = t
        self.time = t

        for meas in self.collection.values():
            meas.time = t
            meas.end_service = t

        self.write_collection()

    def set_collection(self):
        self.collection = {}
        root = ET.parse(self.measure_path).getroot()
        for _measurement in root:
            self.collection[int(_measurement.get('id'))] = ElectricVehicleMeasurement()
            self.collection[int(_measurement.get('id'))].update(_measurement)

    def read_measurements(self):
        root = ET.parse(self.measure_path).getroot()
        for _measurement in root:
            self.collection[int(_measurement.get('id'))].update(_measurement)

    def done(self):
        for meas in self.collection.values():
            if not meas.done:
                return False
        return True

    def write_collection(self, path=None):
        root = ET.Element('measurements')
        for measurement in self.collection.values():
            root.append(measurement.xml_element())
        if path:
            ET.ElementTree(root).write(path)
        else:
            ET.ElementTree(root).write(self.measure_path)

    def observe(self) -> Tuple[net.Network, fleet.Fleet, Dict[int, Tuple[Tuple, Tuple]]]:
        n = net.from_xml(self.network_path, False)
        f = fleet.from_xml(self.fleet_path, True, True, False)
        f.set_network(n)
        self.read_measurements()
        current_routes = {}

        for id_ev, meas in self.collection.items():
            if meas.done:
                continue

            ev = f.vehicles[id_ev]
            current_routes[id_ev] = (ev.route[0], ev.route[1])

            if meas.is_in_node_from:
                (S, L) = ev.route
                k = S.index(meas.node_from)
                S, L = S[k:], L[k:]
                ev.set_route((S, L), meas.end_service, meas.end_soc, meas.end_payload)
                ev.step(n)
                ev.state_reaching[:, 0] = np.asarray([meas.time, meas.end_soc, meas.end_payload])

            else:
                (S, L) = ev.route
                i = S.index(meas.node_to)
                S0, L0 = S[i], L[i]
                t_reach = meas.time + (1 - meas.eta) * n.t(meas.node_from, meas.node_to, meas.time)
                e_reach = meas.soc - 100.*(1 - meas.eta) * n.e(meas.node_from, meas.node_to, meas.payload, ev.weight,
                                                          meas.time)/ev.battery_capacity
                w_reach = meas.payload
                t_leave = t_reach + n.spent_time(meas.node_to, e_reach, L0)
                e_leave = e_reach + L0
                w_leave = meas.payload - n.demand(meas.node_to)
                route = ((S[i:], L[i:]), t_leave, e_leave, w_leave)

                ev.set_route(route[0], route[1], route[2], route[3])
                ev.step(n)
                ev.state_reaching[:, 0] = np.asarray([t_reach, e_reach, w_reach])

            for k, t in enumerate(ev.state_reaching[0, :]):
                if len(ev.route[0]) - k < self.num_of_last_nodes + 1:
                    f.drop_vehicle(id_ev)
                    meas.done = True
                    break
                elif t - meas.time >= self.ga_time + self.offset_time:
                    S, L = ev.route[0][k:], ev.route[1][k:]
                    ev.route = (S, L)
                    ev.state_reaching = ev.state_reaching[:, k:]
                    ev.state_leaving = ev.state_leaving[:, k:]
                    ev.assigned_customers = tuple(i for i in S if f.network.isCustomer(i))
                    break
        now = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        self.write_collection(f'{self.report_folder}{now}.xml')
        return n, f, current_routes
