from res.models.Node import *
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

    def get_energy_consumption(self, payload: float, vehicle_weight: float, time_of_day=None, tAB=None) -> Union[
        float, int]:
        if not self.energy_consumption > 0.:
            return 0.
        v = 60 * self.distance / self.travel_time
        m = vehicle_weight + payload
        g = 127008
        Cr = 1.75
        c1 = 4.575 * 1000 / 3600
        c2 = 1.75
        rho_air = 1225600
        Af = 2.3316e-6
        Cd = 0.28
        eta = 3.17768 * 0.92 * 0.91 * 0.9
        factor_kwh = 100 / (1296 * 3600)
        eAB = factor_kwh * self.distance * (m * g * Cr * (c1 * v + c2) / 1000 + rho_air * Af * Cd * v ** 2 / 2) / eta
        return eAB

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
        return tt

    def get_energy_consumption(self, payload: float, vehicle_weight: float, time_of_day: float, tAB: float = None,
                               breakpoints_info: Tuple[int, int] = None) -> Union[float, int]:
        if self.energy_consumption[0] == 0.:
            return 0.

        while time_of_day >= 1440:
            time_of_day -= 1440

        if tAB is None:
            tAB = self.get_travel_time(time_of_day, breakpoints_info) * 60  # seconds
        else:
            tAB = tAB * 60

        dAB = self.distance * 1000  # meters
        v = dAB / tAB  # m/s
        rho_air = 1.2256  # kg/m^3
        Af = 2.3316  # m^2
        Cd = 0.28
        eta = 0.92 * 0.91 * 0.9

        beta = v ** 2 * rho_air * Af * Cd / 2.
        if breakpoints_info:
            i, j = breakpoints_info
        else:
            i = int(time_of_day / self.sample_time)
            j = i + 1 if i + 1 < len(self.energy_consumption) else 0

        m = 3.6e6 * (self.energy_consumption[j] - self.energy_consumption[i]) / (self.sample_time * 60)  # J/s
        n = 3.6e6 * self.energy_consumption[i] - m * i * self.sample_time * 60  # J
        eAB = m * time_of_day * 60 + n  # J

        alpha = (eAB * eta / dAB - beta) / (vehicle_weight * 1000)
        return ((1000 * (vehicle_weight + payload) * alpha + beta) * dAB / eta) / 3.6e6

    def waiting_time(self, done_time, t_low, payload_after, vehicle_weight) -> Tuple[float, float, float, float]:
        while done_time > 1440:
            done_time -= 1440

        # Waiting time in Sk1
        tt_leaving_now = self.get_travel_time(done_time)
        wt1 = max(0, t_low - tt_leaving_now - done_time + .001)
        ec1 = self.get_energy_consumption(payload_after, vehicle_weight, done_time)

        if wt1 == 0:
            # Do not wait at all because leaving right now accomplishes time window
            return tt_leaving_now, ec1, 0., 0.

        # Waiting time in SK0
        j = int(done_time / self.sample_time) + 1
        wt = self.sample_time * j - done_time

        while wt + self.get_travel_time(done_time + wt) < t_low - done_time:
            j += 1
            wt = self.sample_time * j - done_time

        i = j - 1
        m = (self.travel_time[j] - self.travel_time[i]) / self.sample_time
        n = self.travel_time[i] - m * i * self.sample_time
        val = (t_low - n) / (1 + m) - done_time

        wt0 = max(val + .001, 0) if val < 1440 else max(val + .001 - 1440, 0)
        ec0 = self.get_energy_consumption(payload_after, vehicle_weight, done_time + wt)

        # Choose
        if ec0 >= ec1:
            ec = ec1
            tt = tt_leaving_now
            wt0 = 0.
        else:
            ec = ec0
            tt = self.get_travel_time(done_time + wt0, (i, j))
            wt1 = 0.

        return tt, ec, wt0, wt1

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
