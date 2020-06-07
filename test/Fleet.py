from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

from models.ElectricVehicle import ElectricVehicle
from models.Network import Network
from models.Node import DepotNode, CustomerNode, ChargeStationNode
from models.Edge import Edge
from models.Fleet import Fleet, from_xml

import numpy as np
from numpy import ndarray, zeros, array
import xml.etree.ElementTree as ET

if __name__ == '__main__':
    path = '../notebooks/example.xml'
    fleet_file = from_xml(path)
    # Make an update
    tree = fleet_file.update_from_xml(path, do_network=False)

    routes = {}
    for _vehicle in tree.find('fleet'):
        ev_id = int(_vehicle.get('id'))
        _previous_sequence = _vehicle.find('previous_route')
        if _previous_sequence:
            _critical_point = _vehicle.find('critical_point')
            k = int(_critical_point.get('k'))
            new_route = (tuple(int(x.get('Sk')) for x in _previous_sequence[k:]),
                         tuple(float(x.get('Lk')) for x in _previous_sequence[k:]))
            init_cond = fleet_file.starting_points[ev_id]
            routes[ev_id] = (new_route, init_cond.x1_0, init_cond.x2_0, init_cond.x3_0)

    fleet_file.set_routes_of_vehicles(routes)
    fleet_file.create_optimization_vector()
    feasible, penalization = fleet_file.feasible()
    fleet_file.plot_operation()
    a = 1
