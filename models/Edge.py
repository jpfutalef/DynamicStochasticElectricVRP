from dataclasses import dataclass
from typing import Union, List, Dict
from numpy import ndarray, array, linspace


@dataclass
class Edge:
    node_from: int
    node_to: int
    travel_time: Union[int, float]
    energy_consumption: Union[int, float]

    def get_travel_time(self) -> Union[float, int]:
        return self.travel_time

    def get_energy_consumption(self, payload=0.0) -> Union[float, int]:
        return self.energy_consumption


@dataclass
class DynamicEdge(Edge):
    travel_time: ndarray
    energy_consumption: ndarray
    sample_time: int

    def get_travel_time(self, time_of_day=720.0) -> Union[float, int]:
        return self.travel_time[int(time_of_day/self.sample_time)]

    def get_energy_consumption(self, payload=0.0, time_of_day=720.0) -> Union[float, int]:
        return self.energy_consumption[int(time_of_day/self.sample_time)]


if __name__ == '__main__':
    node_from = 1
    node_to = 2

    # An static Edge
    print(''' ****** Static edge  ****** ''')
    tt = 15
    ec = 18
    edge = Edge(node_from, node_to, tt, ec)
    print('Travel time: %d' % edge.get_travel_time())
    print('Energy consumption: %d' % edge.get_energy_consumption())

    # A dynamic edge
    print(''' ****** Dynamic edge ****** ''')
    sampling_time = 15
    tt = array([15 for x in range(0, 8*60, sampling_time)] +
               [25 for x in range(8*60, 11*60, sampling_time)] +
               [15 for x in range(11*60, 18*60, sampling_time)] +
               [25 for x in range(18*60, 21*60, sampling_time)] +
               [15 for x in range(21*60, 24*60, sampling_time)])
    ec = array([12 for x in range(0, 8 * 60, sampling_time)] +
               [22 for x in range(8 * 60, 11 * 60, sampling_time)] +
               [12 for x in range(11 * 60, 18 * 60, sampling_time)] +
               [22 for x in range(18 * 60, 21 * 60, sampling_time)] +
               [12 for x in range(21 * 60, 24 * 60, sampling_time)])
    dynamic_edge = DynamicEdge(node_from, node_to, tt, ec, sampling_time)

    time_of_day = 12*60
    print('Travel time at TOD %d: %d' %
          (time_of_day, dynamic_edge.get_travel_time(time_of_day)))
    print('Energy consumption at TOD %d: %d'
          % (time_of_day, dynamic_edge.get_energy_consumption(time_of_day=time_of_day)))
