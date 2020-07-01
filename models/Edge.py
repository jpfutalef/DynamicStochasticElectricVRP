from dataclasses import dataclass
from typing import Union, List, Dict
from numpy import ndarray, array, linspace
import xml.etree.ElementTree as ET


@dataclass
class Edge:
    node_from: int
    node_to: int
    travel_time: Union[int, float]
    energy_consumption: Union[int, float]

    def get_travel_time(self, time_of_day=None) -> Union[float, int]:
        return self.travel_time

    def get_energy_consumption(self, payload: float, vehicle_weight: float, time_of_day=None) -> Union[float, int]:
        return self.energy_consumption*(payload + vehicle_weight)/1.52

    def xml_element(self):
        attribs = {str(i): str(j) for i, j in self.__dict__.items()}
        attribs['id'] = str(self.node_to)
        del attribs['node_from'], attribs['node_to']
        element = ET.Element('node_to', attrib=attribs)
        return element


@dataclass
class DynamicEdge:
    node_from: int
    node_to: int
    sample_time: int
    travel_time: ndarray
    energy_consumption: ndarray

    def get_travel_time(self,  time_of_day: float) -> Union[float, int]:
        while time_of_day > 1440:
            time_of_day -= 1440
        return self.travel_time[int(time_of_day/self.sample_time)]

    def get_energy_consumption(self, payload: float, vehicle_weight: float, time_of_day: float) -> Union[float, int]:
        while time_of_day > 1440:
            time_of_day -= 1440
        return self.energy_consumption[int(time_of_day/self.sample_time)]*(payload + vehicle_weight)/1.52

    def xml_element(self):
        attribs = {'id': str(self.node_to)}
        element = ET.Element('node_to', attrib=attribs)

        # Travel time and energy consumption elements
        _tt, _ec = ET.SubElement(element, 'travel_time'), ET.SubElement(element, 'energy_consumption')
        for k, (tt, ec) in enumerate(zip(self.travel_time, self.energy_consumption)):
            attrib_tt = {'time_of_day': str(k*self.sample_time), 'value': str(tt)}
            attrib_ec = {'time_of_day': str(k*self.sample_time), 'value': str(ec)}
            ET.SubElement(_tt, 'breakpoint', attrib=attrib_tt)
            ET.SubElement(_ec, 'breakpoint', attrib=attrib_ec)
        return element

