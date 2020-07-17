from dataclasses import dataclass
from typing import List, Dict
import models.Network as net
import models.OnlineFleet as fleet
import xml.etree.ElementTree as ET


@dataclass
class ElectricVehicleMeasurement:
    id: int
    node_from: int
    node_to: int
    eta: float
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


@dataclass
class Observer:
    collection: Dict[int, ElectricVehicleMeasurement]
    network_path: str
    fleet_path: str
    measure_path: str

    def read_measurements(self):
        root = ET.parse(self.measure_path).getroot()
        for _measurement in root:
            self.collection[int(_measurement.get('id'))].update(_measurement)

    def observe(self):
        n = net.from_xml(self.network_path)
        f = fleet.from_xml(self.fleet_path,)



