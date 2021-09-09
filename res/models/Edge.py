from dataclasses import dataclass
from typing import Union, Dict, Tuple
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt


def fix_tod(tod: float):
    while tod > 86400:
        tod -= 86400
    return tod


def get_breakpoint_indices(t: float, dt: float, array: np.ndarray):
    i = int(t / dt)
    j = i + 1 if i + 1 < len(array) else 0
    return i, j


def get_affine_parameters(x0, x1, y0, y1):
    m = (y1 - y0) / (x1 - x0)
    n = y0 - m * x0
    return m, n


def interpolate(t: float, dt: float, array: Union[np.ndarray, list]):
    i = int(t / dt)
    j = i + 1 if i + 1 < len(array) else 0
    t0, y0 = i * dt, array[i]
    t1, y1 = (i + 1) * dt, array[j]
    m, n = get_affine_parameters(t0, t1, y0, y1)
    #plt.plot(array)
    #plt.plot(t / dt, m * t + n, "*")
    #plt.show()
    return m * t + n


@dataclass
class Edge:
    node_from: int
    node_to: int
    distance: float
    velocity: float

    def get_velocity(self):
        return self.velocity

    def xml_element(self):
        attribs = {str(i): str(j) for i, j in self.__dict__.items()}
        attribs['id'] = str(self.node_to)
        del attribs['node_from'], attribs['node_to']
        element = ET.Element('node_to', attrib=attribs)
        return element

    @classmethod
    def from_xml_element(cls, element: ET.Element):
        return


@dataclass
class DynamicEdge:
    node_from: int
    node_to: int
    length: float
    sample_time: float
    velocity: np.ndarray
    road_length_profile: np.ndarray
    road_inclination_profile: np.ndarray
    velocity_unit: str = 'm/s'
    road_sin_length: float = None
    road_cos_length: float = None
    type: str = None

    def __post_init__(self):
        self.road_cos_length = np.dot(np.cos(self.road_inclination_profile), self.road_length_profile)
        self.road_sin_length = np.dot(np.sin(self.road_inclination_profile), self.road_length_profile)
        self.type = self.__class__.__name__

    def get_velocity(self, time_of_day: float) -> float:
        time_of_day = fix_tod(time_of_day)
        v = interpolate(time_of_day, self.sample_time, self.velocity)
        return v

    def xml_element(self):
        attribs = {'id': str(self.node_to), 'length': str(self.length), 'type': self.type,
                   'sample_time': str(self.sample_time), 'velocity_unit': str(self.velocity_unit)}
        element = ET.Element('node_to', attrib=attribs)
        _v = ET.SubElement(element, 'velocity')
        for k, v in enumerate(self.velocity):
            attr = {'time_of_day': str(k * self.sample_time), 'value': str(v)}
            ET.SubElement(_v, 'breakpoint', attrib=attr)
        _arc_profile = ET.SubElement(element, 'arc_profile')
        for length, inclination in zip(self.road_length_profile, self.road_inclination_profile):
            attr = {'length': str(length), 'inclination': str(inclination)}
            ET.SubElement(_arc_profile, 'road', attrib=attr)
        return element

    @classmethod
    def from_xml_element(cls, element: ET.Element, node_from: int):
        node_to = int(element.get('id'))
        length = float(element.get('length'))
        sample_time = float(element.get('sample_time'))
        velocity_unit = str(element.get('velocity_unit'))
        velocity = np.array([float(_bp.get('value')) for _bp in element.find('velocity')])

        _road_profile = element.find('arc_profile')
        distance_profile = np.array([float(_bp.get('length')) for _bp in _road_profile])
        inclination_profile = np.array([float(_bp.get('inclination')) for _bp in _road_profile])

        return cls(node_from, node_to, length, sample_time, velocity, distance_profile, inclination_profile,
                   velocity_unit)

    def plot_profile(self):
        fig = plt.figure()
        plt.plot(np.arange(0, 86400, 1800), self.velocity)
        plt.xlabel("TOD")
        plt.ylabel("Velocity [m/s]")
        return fig


@dataclass
class GaussianEdge(DynamicEdge):
    velocity_deviation: np.ndarray = None
    velocity_source: np.ndarray = None
    velocity_deviation_source: np.ndarray = None

    def __post_init__(self):
        super(GaussianEdge, self).__post_init__()
        if self.velocity_deviation is None:
            self.velocity_deviation = np.zeros_like(self.velocity)
        self.velocity_source = np.copy(self.velocity)
        self.velocity_deviation_source = np.copy(self.velocity_deviation)

    def disturb(self, std_gain: float = 1.0):
        self.velocity = np.random.normal(self.velocity_source, std_gain * self.velocity_deviation_source)

    def get_velocity(self, time_of_day: float) -> Tuple[float, float]:
        time_of_day = fix_tod(time_of_day)
        mu = interpolate(time_of_day, self.sample_time, self.velocity)
        sigma = interpolate(time_of_day, self.sample_time, self.velocity_deviation)
        return mu, sigma

    def xml_element(self):
        element = super(GaussianEdge, self).xml_element()
        _v = element.find('velocity')
        for _bp, v_std in zip(_v, self.velocity_deviation):
            _bp.set('deviation', str(v_std))
        return element

    @classmethod
    def from_xml_element(cls, element: ET.Element, node_from: int):
        node_to = int(element.get('id'))
        length = float(element.get('length'))
        sample_time = float(element.get('sample_time'))
        velocity_unit = str(element.get('velocity_unit'))

        _v = element.find('velocity')
        velocity = np.array([float(_bp.get('value')) for _bp in _v])
        velocity_deviation = np.array([float(_bp.get('deviation')) for _bp in _v])

        _road_profile = element.find('arc_profile')
        length_profile = np.array([float(_bp.get('length')) for _bp in _road_profile])
        inclination_profile = np.array([float(_bp.get('inclination')) for _bp in _road_profile])

        return cls(node_from, node_to, length, sample_time, velocity, length_profile, inclination_profile,
                   velocity_unit, velocity_deviation=velocity_deviation)

    def plot_profile(self):
        fig = plt.figure()
        plt.plot(np.arange(0, 86400, 1800), self.velocity, marker='', markersize=4, color='k', label='Average')
        plt.fill_between(np.arange(0, 86400, 1800), self.velocity + 3*self.velocity_deviation,
                         self.velocity - 3*self.velocity_deviation, alpha=.2, color="black",
                         label='99.7% confidence interval')
        plt.xlabel("TOD")
        plt.ylabel("Velocity [m/s]")
        return fig


def from_xml_element(element: ET.Element, node_from: int):
    t = element.get('type')
    cls = globals()[t]
    return cls.from_xml_element(element, node_from)
