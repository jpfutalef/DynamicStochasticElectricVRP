from dataclasses import dataclass
from typing import List, Dict, Tuple
import models.Network as net
import models.OnlineFleet as fleet
import xml.etree.ElementTree as ET


@dataclass
class ElectricVehicleMeasurement:
    id: int
    node_from: int
    node_to: int
    eta: float
    ti: float
    tf: float
    is_in_node_from: bool
    x1: float
    x2: float
    x3: float
    done: bool

    def xml_element(self) -> [ET.Element]:
        return ET.Element(f'{self.id}', attrib={str(i): str(j) for i, j in self.__dict__.items()})

    def update(self, etree: ET.Element):
        for key, attrib in etree.attrib.items():
            self.__dict__[key] = attrib


def create_measurement_files(f: fleet.Fleet):
    root = ET.Element('measurements')
    for ev in f.vehicles.values():
        m = ElectricVehicleMeasurement(ev.id, ev.route[0][0], ev.route[0][1], 0., True, ev.x1_0, ev.x1_0, ev.x2_0,
                                       ev.x3_0, False)
        root.append(m.xml_element())


@dataclass
class Observer:
    collection: Dict[int, ElectricVehicleMeasurement]
    network_path: str
    fleet_path: str
    measure_path: str
    ga_time: float
    offset_time: float

    def read_measurements(self):
        root = ET.parse(self.measure_path).getroot()
        for _measurement in root:
            self.collection[int(_measurement.get('id'))].update(_measurement)

    def observe(self) -> Tuple[net.Network, fleet.Fleet]:
        n = net.from_xml(self.network_path)
        f = fleet.from_xml(self.fleet_path)
        self.read_measurements()

        for id_ev, meas in self.collection.items():
            ev = f.vehicles[id_ev]
            if meas.is_in_node_from:
                route = (ev.route, meas.x1L, meas.x2L, meas.x3L)
            else:
                (S, L) = ev.route
                i = S.index(meas.node_to)
                S0, L0 = S[i], L[i]
                t_reach = meas.x1 + (1-meas.eta)*n.t(meas.node_from, meas.node_to, meas.x1)
                e_reach = meas.x2 - (1-meas.eta)*n.e(meas.node_from, meas.node_to, meas.x3, ev.weight, meas.x1)
                t_leave = t_reach + n.spent_time(meas.node_to, e_reach, L0)
                e_leave = e_reach + L0
                w_leave = meas.x3 - n.demand(meas.node_to)
                route = ((S[i:], L[i:]), t_leave, e_leave, w_leave)

            ev.set_route(route[0], route[1], route[2], route[3])
            ev.step(n)

            for k, t in enumerate(ev.state_reaching[0, :]):
                if t - meas.x1 >= self.ga_time + self.offset_time:
                    pass
        return n, f
