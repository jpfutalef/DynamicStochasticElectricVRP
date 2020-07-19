import xml.etree.ElementTree as ET

from dataclasses import dataclass
from typing import Union, NamedTuple, List, Dict, Tuple


class CriticalPoint(NamedTuple):
    k: int
    waiting_time: float
    x1: float
    x2: float
    x3: float


class EVBuffer(NamedTuple):
    previous_sequence: Union[List[Tuple[int, Union[int, float]]], None]
    critical_point: Union[CriticalPoint, None]


class EVRPBuffer(NamedTuple):
    network_data: Dict[int, Dict[str, float]]
    fleet_data: Dict[int, EVBuffer]


@dataclass
class XMLParserRealtime:
    path: str

    def read(self, verbose=False) -> EVRPBuffer:
        # Open XML file
        tree: ET = ET.parse(self.path)
        _network: ET = tree.find('network')
        _fleet: ET = tree.find('fleet')

        # [START Update fleet data]
        for _vehicle in _fleet:
            _prev_seq = _vehicle.find('previous_sequence')

            # Case 1: There is a possibility to improve route
            if len(_prev_seq) > 0:
                id_vehicle = int(_vehicle.get('id'))
                _critical_point = _vehicle.find('critical_point')
                cp_info = _critical_point.attrib

                k = int(cp_info['k'])
                waiting_time = float(cp_info['waiting_time'])
                x1 = float(cp_info['x1'])
                x2 = float(cp_info['x2'])
                x3 = float(cp_info['x3'])

                data: EVRPData = EVRPData(k, waiting_time, x1, x2, x3)

                # self.update_vehicle(data)

            # Case 2: Do not route this vehicle
            else:
                pass

        # [END Update fleet data]

        # [START Edge data]
        _edges = _network.find('edges')
        travel_time = {}
        energy_consumption = {}

        for i, nodeFrom in enumerate(_edges):
            tt_dict = travel_time[i] = {}
            ec_dict = energy_consumption[i] = {}
            for j, nodeTo in enumerate(nodeFrom):
                # The following assumes that travel times and energy
                # consumption are deterministic values
                tt_dict[j] = float(nodeTo.get('travel_time'))
                ec_dict[j] = float(nodeTo.get('energy_consumption'))

        self.network.set_travel_time(travel_time)
        self.network.set_energy_consumption(energy_consumption)
        # [END Edge data]

    def write(self):
        pass


class RawObserver:
    def __init__(self):
        pass


class EVRPObserver:
    def __init__(self):
        pass
