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

    def get_energy_consumption(self, payload) -> Union[float, int]:
        return payload*self.energy_consumption


@dataclass
class DynamicEdge:
    node_from: int
    node_to: int
    travel_time: ndarray
    energy_consumption: ndarray
    sample_time: int

    def get_travel_time(self, time_of_day) -> Union[float, int]:
        return self.travel_time[int(time_of_day/self.sample_time)]

    def get_energy_consumption(self, payload, time_of_day) -> Union[float, int]:
        return payload*self.energy_consumption[int(time_of_day/self.sample_time)]
