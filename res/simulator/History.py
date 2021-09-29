import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, NamedTuple, Union, Dict, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import res.models.Fleet as Fleet
import res.models.Network as Network
import res.dispatcher.Dispatcher as Dispatcher
import res.tools.IOTools as IOTools


@dataclass
class ConstraintViolation:
    type: str = 'generic'
    constraint_value: Union[int, float] = None
    real_value: Union[int, float] = None
    where: Union[int, float] = None
    when: str = None

    def xml_element(self):
        attrib = {str(key): str(val) for key, val in self.__dict__.items()}
        return ET.Element('violated_constraint', attrib)

    @classmethod
    def from_xml_element(cls, element: ET.Element):
        the_type = str(element.get('type'))
        constraint_value = float(element.get('constraint_value'))
        real_value = float(element.get('real_value'))
        where = int(element.get('where'))
        when = str(element.get('when'))
        return cls(the_type, constraint_value, real_value, where, when)


@dataclass
class NodeEvent:
    arriving: bool = True
    time: Union[int, float] = None
    soc: Union[int, float] = None
    payload: Union[int, float] = None
    where: Union[int, float] = None
    pre_service_waiting_time: float = 0.
    post_service_waiting_time: float = 0.

    def get_state(self):
        return np.array([self.time, self.soc, self.payload])

    def xml_element(self):
        attrib = {str(key): str(val) for key, val in self.__dict__.items()}
        return ET.Element('event', attrib)

    @classmethod
    def from_xml_element(cls, element: ET.Element):
        arriving = True if element.get('arriving') == 'True' else False
        time = float(element.get('time'))
        soc = float(element.get('soc'))
        payload = float(element.get('payload'))
        where = int(element.get('where'))
        pre_service_waiting_time = float(element.get('pre_service_waiting_time'))
        post_service_waiting_time = float(element.get('post_service_waiting_time'))
        return cls(arriving, time, soc, payload, where, pre_service_waiting_time, post_service_waiting_time)


@dataclass
class EVHistory:
    id: int
    travelled_time: float = 0.
    consumed_energy: float = 0.
    recharging_time: float = 0.
    recharging_cost: float = 0.
    violated_constraints: List[ConstraintViolation] = None
    arriving_events: List[NodeEvent] = None
    departing_events: List[NodeEvent] = None

    def __post_init__(self):
        if self.violated_constraints is None:
            self.violated_constraints = []
        if self.arriving_events is None:
            self.arriving_events = []
        if self.departing_events is None:
            self.departing_events = []

    def add_violation(self, violated_constraint: ConstraintViolation):
        self.violated_constraints.append(violated_constraint)

    def add_event(self, event: NodeEvent):
        if event.arriving:
            self.arriving_events.append(event)
        else:
            self.departing_events.append(event)

    def draw_events(self, network: Network.Network, fleet: Fleet.Fleet, **kwargs) -> Tuple[plt.Axes, Tuple]:
        # Setup
        S_dep = [e.where for e in self.departing_events]
        S_arr = [e.where for e in self.arriving_events]

        add1 = False if len(S_dep) - len(S_arr) else True
        num_events = len(S_dep) + 1 if add1 else len(S_dep)

        departing_state = np.zeros((3, num_events))
        for k, e in enumerate(self.departing_events):
            departing_state[:, k] = e.get_state()
        if add1:
            departing_state[:, -1] = self.arriving_events[-1].get_state()

        arriving_state = np.zeros((3, num_events))
        arriving_state[:, 0] = departing_state[:, 0]
        for k, e in enumerate(self.arriving_events):
            arriving_state[:, k + 1] = e.get_state()

        S = S_dep + [S_arr[-1]] if add1 else S_dep

        # Create containers
        X, XX, times, soc, payload = [], [], [], [], []
        for k, Sk in enumerate(S):
            X += [k]
            XX += [k] + [k]
            times += [arriving_state[0, k], departing_state[0, k]]
            soc += [arriving_state[1, k], departing_state[1, k]]
            payload += [arriving_state[2, k], departing_state[2, k]]

        # Time windows
        twu = network.time_window_upp
        twl = network.time_window_low
        tws = np.array([[(twu(i) + twl(i)) / 2, (twu(i) - twl(i)) / 2] if network.is_customer(i) else [-1, -1] for i in S])

        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, **kwargs)

        # Time figure
        for k, (Sk, Xk) in enumerate(zip(S, X)):
            if network.is_customer(Sk):
                ax1.errorbar(Xk, tws[k, 0], tws[k, 1], ecolor='black', fmt='none', capsize=6, elinewidth=1,)
            ax1.annotate(str(Sk), (Xk, arriving_state[0, k] - 100))
        ax1.plot(XX, times)
        ax1.set_xlabel('Stop')
        ax1.set_ylabel('Time of day [s]')
        ax1.set_title(f'Time EV {self.id}')

        # SOC figure
        ax2.plot(XX, soc)
        ev = fleet.vehicles[self.id]
        ax2.axhline(ev.alpha_down, linestyle='--', color='black', label='SOH policy')
        ax2.axhline(ev.alpha_up, linestyle='--', color='black', label=None)
        ax2.fill_between([-1, len(S)], ev.alpha_down, ev.alpha_up, color='lightgrey', alpha=.35)
        [ax2.annotate(str(Sk), (Xk, arriving_state[1, k])) for k, (Sk, Xk) in enumerate(zip(S, X))]
        ax2.set_xlabel('Stop')
        ax2.set_ylabel('SOC [%]')
        ax2.set_title(f'SOC EV {self.id}')
        ax2.set_ylim((0, 100))
        ax2.set_xlim((-.5, len(S) - .5))

        # Payload figure
        ax3.plot(XX, payload)
        [ax3.annotate(str(Sk), (Xk, arriving_state[2, k])) for k, (Sk, Xk) in enumerate(zip(S, X))]
        ax3.set_xlabel('Stop')
        ax3.set_ylabel('Payload [kg]')
        ax3.set_title(f'Payload EV {self.id}')

        # Final figure setups
        fig.tight_layout()

        # Info
        info = (self.id, S, arriving_state, departing_state)

        return fig, info

    def to_pandas(self) -> pd.DataFrame:
        return

    def xml_element(self) -> ET.Element:
        skip_in_ev = ['violated_constraints', 'departing_events', 'arriving_events']
        attrib = {key: str(attr) for key, attr in self.__dict__.items() if key not in skip_in_ev}

        _element = ET.Element('vehicle', attrib=attrib)
        _violated_constraints = ET.SubElement(_element, 'violated_constraints')
        _arriving_events = ET.SubElement(_element, 'arriving_events')
        _departing_events = ET.SubElement(_element, 'departing_events')

        for violated_constraint in self.violated_constraints:
            _violated_constraints.append(violated_constraint.xml_element())

        for event in self.arriving_events:
            _arriving_events.append(event.xml_element())

        for event in self.departing_events:
            _departing_events.append(event.xml_element())

        return _element

    @classmethod
    def from_xml_element(cls, element: ET.Element):
        the_id = int(element.get('id'))
        travelled_time = float(element.get('travelled_time'))
        consumed_energy = float(element.get('consumed_energy'))
        recharging_time = float(element.get('recharging_time'))
        recharging_cost = float(element.get('recharging_cost'))
        violated_constraints = [ConstraintViolation.from_xml_element(e) for e in element.find('violated_constraints')]
        arriving_events = [NodeEvent.from_xml_element(e) for e in element.find('arriving_events')]
        departing_events = [NodeEvent.from_xml_element(e) for e in element.find('departing_events')]
        return cls(the_id, travelled_time, consumed_energy, recharging_time, recharging_cost, violated_constraints,
                   arriving_events, departing_events)


@dataclass
class FleetHistory:
    travelled_time: float = 0.
    consumed_energy: float = 0.
    recharging_time: float = 0.
    recharging_cost: float = 0.
    violated_constraints: List[ConstraintViolation] = None
    vehicles_history: Dict[int, EVHistory] = None

    def __post_init__(self):
        if self.violated_constraints is None:
            self.violated_constraints = []
        if self.vehicles_history is None:
            self.vehicles_history = {}

    def update_travelled_time(self, id_ev: int, increment: float):
        self.vehicles_history[id_ev].travelled_time += increment
        self.travelled_time += increment

    def update_recharging_time(self, id_ev: int, increment: float):
        self.vehicles_history[id_ev].recharging_time += increment
        self.recharging_time += increment

    def update_recharging_cost(self, id_ev: int, increment: float):
        self.vehicles_history[id_ev].recharging_cost += increment
        self.recharging_cost += increment

    def update_consumed_energy(self, id_ev: int, increment: float):
        self.vehicles_history[id_ev].consumed_energy += increment
        self.consumed_energy += increment

    def add_vehicle_constraint_violation(self, id_ev: int, violation: ConstraintViolation):
        self.vehicles_history[id_ev].add_violation(violation)

    def add_fleet_constraint_violation(self, violation: ConstraintViolation):
        self.violated_constraints.append(violation)

    def add_vehicle_event(self, id_ev: int, event: NodeEvent):
        self.vehicles_history[id_ev].add_event(event)

    def draw_events(self, n: Network.Network, fleet: Fleet.Fleet, **kwargs):
        vehicles_figs, vehicles_info = [], []
        for ev_history in self.vehicles_history.values():
            drawing = ev_history.draw_events(n, fleet, **kwargs)
            vehicles_figs.append(drawing[0])
            vehicles_info.append(drawing[1])
        return vehicles_figs, vehicles_info

    @classmethod
    def create_from_routes(cls, routes: Dispatcher.RouteDict):
        vehicles_history = {}
        for id_ev in routes.keys():
            vehicles_history[id_ev] = EVHistory(id_ev)
        return cls(vehicles_history=vehicles_history)

    @classmethod
    def from_fleet_size(cls, size: int):
        vehicles_history = {}
        for id_ev in range(size):
            vehicles_history[id_ev] = EVHistory(id_ev)
        return cls(vehicles_history=vehicles_history)

    @classmethod
    def from_fleet(cls, fleet: Fleet.Fleet):
        vehicles_history = {}
        for id_ev in fleet.vehicles.keys():
            vehicles_history[id_ev] = EVHistory(id_ev)
        return cls(vehicles_history=vehicles_history)

    @classmethod
    def from_xml_element(cls, element: ET.Element):
        _fleet = element.find('fleet')
        travelled_time = float(_fleet.get('travelled_time'))
        consumed_energy = float(_fleet.get('consumed_energy'))
        recharging_time = float(_fleet.get('recharging_time'))
        recharging_cost = float(_fleet.get('recharging_cost'))
        violated_constraints = [ConstraintViolation.from_xml_element(e) for e in _fleet.find('violated_constraints')]

        _vehicles = element.iter('vehicle')
        vehicles_history = {}
        for _vehicle in _vehicles:
            the_id = int(_vehicle.get('id'))
            vehicles_history[the_id] = EVHistory.from_xml_element(_vehicle)
        return cls(travelled_time, consumed_energy, recharging_time, recharging_cost, violated_constraints,
                   vehicles_history)

    @classmethod
    def from_xml(cls, filepath: Path):
        root = ET.parse(filepath).getroot()
        return cls.from_xml_element(root)

    def to_pandas(self) -> pd.DataFrame:
        return

    def save(self, filepath: Path, write_pretty: bool = False):
        root = ET.Element('history')
        skip = ['violated_constraints', 'vehicles_history']
        fleet_attr = {str(key): str(val) for key, val in self.__dict__.items() if key not in skip}
        _fleet_history = ET.SubElement(root, 'fleet', attrib=fleet_attr)
        _violated_constraints = ET.SubElement(_fleet_history, 'violated_constraints')

        for violated_constraint in self.violated_constraints:
            _violated_constraints.append(violated_constraint.xml_element())

        for ev_history in self.vehicles_history.values():
            root.append(ev_history.xml_element())

        if write_pretty:
            IOTools.write_pretty_xml(filepath, root)
        else:
            ET.ElementTree(root).write(filepath)
