from dataclasses import dataclass
from typing import Tuple, Union

import xml.etree.ElementTree as ET


@dataclass
class NetworkNode:
    """
    A generic node.
    """
    id: int
    spent_time: float = 0.0
    demand: float = 0.0
    pos_x: float = 0.0
    pos_y: float = 0.0

    def spentTime(self, p, q):
        """
        The time an EV in this node.
        :param q: EV SOC when it arrives to this node
        :param p: EV SOC increment in this node
        :return: Time the EV spends here
        """
        return self.spent_time

    def isChargeStation(self):
        return False

    def isDepot(self):
        return False

    def isCustomer(self):
        return False

    def xml_element(self):
        attribs = {str(i): str(j) for i, j in self.__dict__.items()}
        element = ET.Element('node', attrib=attribs)
        return element


@dataclass
class DepotNode(NetworkNode):
    type: int = 0

    def isDepot(self):
        return True


@dataclass
class CustomerNode(NetworkNode):
    time_window_low: float = 0.0
    time_window_upp: float = 239.0
    type: int = 1

    def isCustomer(self):
        return True


@dataclass
class ChargeStationNode(NetworkNode):
    capacity: int = 4
    time_points: Tuple[float, ...] = (0.0, 50.0, 120.0)
    soc_points: Tuple[float, ...] = (0.0, 80.0, 100.0)
    technology: int = 1
    type: int = 2

    def calculateTimeSpent(self, init_soc, end_soc, eta=1.0):
        t_points, soc_points = self.time_points, self.soc_points

        end_time = t_points[-1] * 10. if end_soc > 100. else 0.
        init_time = -t_points[-1] * 10. if end_soc < 0. else 0.

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

        return end_time - init_time

    def spentTime(self, init_soc, increment):
        return self.calculateTimeSpent(init_soc, init_soc + increment)

    def isChargeStation(self):
        return True

    def xml_element(self):
        attribs = {str(i): str(j) for i, j in self.__dict__.items()}
        del attribs['spent_time'], attribs['time_points'], attribs['soc_points']
        element = ET.Element('node', attrib=attribs)
        return element