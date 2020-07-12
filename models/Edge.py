from models.Node import *
from dataclasses import dataclass


@dataclass
class Edge:
    node_from: int
    node_to: int
    travel_time: Union[int, float]
    energy_consumption: Union[int, float]
    distance: float

    def get_travel_time(self, time_of_day=None) -> Union[float, int]:
        return self.travel_time

    def get_energy_consumption(self, payload: float, vehicle_weight: float, time_of_day=None) -> Union[float, int]:
        # return self.energy_consumption*(payload + vehicle_weight)/1.52
        dAB = self.distance
        tAB = self.travel_time
        rho_air = 1225600
        Af = 2.3316 / 1000000
        Cd = 0.28
        beta = (dAB / tAB) ** 3 * rho_air * Af * Cd / 2.
        eAB = self.energy_consumption
        alpha = (eAB * .92 * .91 * .9 / tAB - beta) / vehicle_weight
        return ((payload + vehicle_weight) * alpha + beta) * tAB / (.92 * .91 * .9)

    def waiting_time(self, done_time, t_low, payload_after, vehicle_weight) -> Tuple[float, float, float]:
        tt, ec = self.travel_time, self.get_energy_consumption(payload_after, vehicle_weight)
        wt = max(0, t_low - tt - done_time)
        return tt, ec, wt

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
    travel_time: np.ndarray
    energy_consumption: np.ndarray
    distance: float

    def get_travel_time(self, time_of_day: float, breakpoints_info: Tuple[int, int] = None) -> Union[float, int]:
        while time_of_day >= 1440:
            time_of_day -= 1440
        if breakpoints_info:
            i, j = breakpoints_info
        else:
            i = int(time_of_day / self.sample_time)
            j = i + 1 if i + 1 < len(self.travel_time) else 0
        m = (self.travel_time[j] - self.travel_time[i]) / self.sample_time
        n = self.travel_time[i] - m * i * self.sample_time
        tt = m * time_of_day + n
        # return self.travel_time[int(time_of_day/self.sample_time)]
        return tt

    def get_energy_consumption(self, payload: float, vehicle_weight: float, time_of_day: float,
                               breakpoints_info: Tuple[int, int] = None, tAB: float = None) -> Union[float, int]:
        while time_of_day >= 1440:
            time_of_day -= 1440
        # return self.energy_consumption[int(time_of_day/self.sample_time)]*(payload + vehicle_weight)/1.52
        dAB = self.distance
        if tAB is None:
            tAB = self.get_travel_time(time_of_day, breakpoints_info)
        rho_air = 1225600
        Af = 2.3316e-6
        Cd = 0.28
        beta = (dAB / tAB) ** 3 * rho_air * Af * Cd / 2. if tAB > 0.0 else 0.0

        if breakpoints_info:
            i, j = breakpoints_info
        else:
            i = int(time_of_day / self.sample_time)
            j = i + 1 if i + 1 < len(self.energy_consumption) else 0
        m = (self.energy_consumption[j] - self.energy_consumption[i]) / self.sample_time
        n = self.energy_consumption[i] - m * i * self.sample_time
        eAB = m * time_of_day + n
        alpha = (eAB*.92*.91*.9/tAB-beta)/vehicle_weight if tAB > 0.0 else 0.0
        return ((payload+vehicle_weight)*alpha + beta)*tAB/(.92*.91*.9)

    def waiting_time(self, done_time, t_low, payload_after, vehicle_weight) -> Tuple[float, float, float]:
        while done_time > 1440:
            done_time -= 1440
        i = int(done_time / self.sample_time) + 1
        wt = self.sample_time * i - done_time
        while wt + self.get_travel_time(done_time + wt) < t_low - done_time:
            i += 1
            wt = self.sample_time * i - done_time

        i = i if i < len(self.energy_consumption) else 0
        j = i - 1
        m = (self.travel_time[j] - self.travel_time[i]) / self.sample_time
        n = self.travel_time[i] - m * i * self.sample_time
        wt = max((t_low - n) / (1 + m) - done_time, 0)

        tt = self.get_travel_time(done_time + wt, (i, j))
        ec = self.get_energy_consumption(payload_after, vehicle_weight, done_time + wt, (i, j))
        return tt, ec, wt

    def xml_element(self):
        attribs = {'id': str(self.node_to), 'distance': str(self.distance)}
        element = ET.Element('node_to', attrib=attribs)

        # Travel time and energy consumption elements
        _tt, _ec = ET.SubElement(element, 'travel_time'), ET.SubElement(element, 'energy_consumption')
        for k, (tt, ec) in enumerate(zip(self.travel_time, self.energy_consumption)):
            attrib_tt = {'time_of_day': str(k * self.sample_time), 'value': str(tt)}
            attrib_ec = {'time_of_day': str(k * self.sample_time), 'value': str(ec)}
            ET.SubElement(_tt, 'breakpoint', attrib=attrib_tt)
            ET.SubElement(_ec, 'breakpoint', attrib=attrib_ec)
        return element
