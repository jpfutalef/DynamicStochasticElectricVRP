import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np


@dataclass
class BaseNode:
    """
    A generic node.
    """
    id: int
    spent_time: float = 0.0
    demand: float = 0.0
    pos_x: float = 0.0
    pos_y: float = 0.0
    time_window_low: float = -np.infty
    time_window_upp: float = np.infty
    type: str = None

    def __post_init__(self):
        self.type = self.__class__.__name__

    def service_time(self, init_soc, soc_increment, eta=None):
        """
        Time the EV stays in this node.
        @param init_soc: SOC value when the EV arrives
        @param soc_increment: SOC value when the EV leaves
        @param eta: battery SOH factor between 0 (degraded) and 1 (fully healthy)
        @return: service time value
        """
        return self.spent_time

    def is_charge_station(self):
        return False

    def is_depot(self):
        return False

    def is_customer(self):
        return False

    def xml_element(self):
        attribs = {str(i): str(j) for i, j in self.__dict__.items()}
        element = ET.Element('node', attrib=attribs)
        return element


@dataclass
class DepotNode(BaseNode):
    def is_depot(self):
        return True

    @classmethod
    def from_xml_element(cls, element: ET.Element):
        node_id = int(element.get('id'))
        pos_x, pos_y = float(element.get('pos_x')), float(element.get('pos_y'))
        node = cls(node_id, pos_x=pos_x, pos_y=pos_y)
        return node


@dataclass
class CustomerNode(BaseNode):
    def is_customer(self):
        return True

    @classmethod
    def from_xml_element(cls, element: ET.Element):
        node_id = int(element.get('id'))
        spent_time = float(element.get('spent_time'))
        time_window_low = float(element.get('time_window_low'))
        time_window_upp = float(element.get('time_window_upp'))
        demand = float(element.get('demand'))
        pos_x, pos_y = float(element.get('pos_x')), float(element.get('pos_y'))
        node = cls(node_id, spent_time, demand, pos_x, pos_y, time_window_low, time_window_upp)
        return node


@dataclass
class ChargingStationNode(BaseNode):
    capacity: Union[int, float] = np.infty
    time_points: Tuple[float, ...] = (0.0, 0.0)
    soc_points: Tuple[float, ...] = (0.0, 100.)
    technology: int = 1
    technology_name: str = 'default'
    price: float = 0.

    def is_charge_station(self):
        return True

    def service_time(self, init_soc, increment, eta: float = None):
        if eta:
            t_points, soc_points = self.time_points, np.array(self.soc_points)
            soc_points[1:-1] = eta * soc_points[1:-1]
        else:
            t_points, soc_points = self.time_points, self.soc_points
        end_soc = init_soc + increment
        init_time = t_points[0]
        end_time = t_points[-1]

        for t0, t1, y0, y1 in zip(t_points[:-1], t_points[1:], soc_points[:-1], soc_points[1:]):
            # find time where operation begins
            if y0 <= init_soc <= y1:
                m = (y1 - y0) / (t1 - t0)
                n = y1 - m * t1
                init_time = (init_soc - n) / m
            # find time where operation ends
            if y0 <= end_soc <= y1:
                m = (y1 - y0) / (t1 - t0)
                n = y1 - m * t1
                end_time = (end_soc - n) / m
                break
        init_time = init_time if 0. <= end_soc <= 100. else -1000*abs(init_time)
        end_time = end_time if 0. <= end_soc <= 100. else 1000*abs(end_time)
        return end_time - init_time

    def set_technology(self, tech_dict):  # TODO use kwargs?
        self.time_points = tuple(tech_dict['description'].keys())
        self.soc_points = tuple(tech_dict['description'].values())
        self.technology = tech_dict['technology']
        self.technology_name = tech_dict['technology_name']
        self.price = tech_dict['price']

    def xml_element(self):
        dont_include = ['spent_time', 'time_points', 'soc_points', 'technology', 'technology_name']
        attribs = {str(i): str(j) for i, j in self.__dict__.items() if i not in dont_include}
        element = ET.Element('node', attrib=attribs)
        _technology = ET.SubElement(element, 'charging_function', {'technology': str(self.technology),
                                                                   'technology_name': str(self.technology_name)})
        for time, soc in zip(self.time_points, self.soc_points):
            attr = {'time': str(time), 'soc': str(soc)}
            _technology.append(ET.Element('breakpoint', attr))
        return element

    @classmethod
    def from_xml_element(cls, element: ET.Element):  # TODO define in parent class and use super()
        node_id = int(element.get('id'))
        pos_x, pos_y = float(element.get('pos_x')), float(element.get('pos_y'))
        capacity = int(element.get('capacity'))
        price = float(element.get('price'))

        _charging_function = element.find('charging_function')
        technology_name = str(_charging_function.get('technology_name'))
        technology = int(_charging_function.get('technology'))
        time_points = tuple([float(bp.get('time')) for bp in _charging_function])
        soc_points = tuple([float(bp.get('soc')) for bp in _charging_function])

        node = cls(node_id, pos_x=pos_x, pos_y=pos_y, capacity=capacity, time_points=time_points, soc_points=soc_points,
                   technology=technology, technology_name=technology_name, price=price)
        return node


def from_xml_element(element: ET.Element):
    t = element.get('type')
    cls = globals()[t]
    return cls.from_xml_element(element)
