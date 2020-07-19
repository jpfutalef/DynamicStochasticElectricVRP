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
            if meas.is_in_node_from:
                if n.isCustomer(meas.node_from):
                    pass
                elif n.isChargingStation(meas.node_from):
                    pass
            else:
                pass
                
        return n, f
