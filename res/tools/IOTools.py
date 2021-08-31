import xml.dom.minidom
import xml.etree.ElementTree as ET
from typing import Tuple, Dict, Union
import os
from pathlib import Path

"""
XML read/out tools
"""


def read_write_pretty_xml(path):
    xml_pretty = xml.dom.minidom.parse(path).toprettyxml()
    with open(path, 'w') as file:
        file.write(xml_pretty)


def write_pretty_xml(filepath: Path, tree: ET.Element):
    s = ET.tostring(tree)
    pretty = xml.dom.minidom.parseString(s).toprettyxml()
    with open(filepath, 'w') as file:
        file.write(pretty)


"""
Folder handlers
"""


def create_folder(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


"""
FRONT-END TYPES
- RouteVector: is a tuple that contains three more tuples with the following information for a single EV:
    [0] The sequence of node (S) 
    [1] The SOC increments (L) 
    [2] The waiting times after the service (w1)
    
- RouteDict: is a dictionary containing EVs' routes. The key is the id of an EV, whereas the values are RouteVector's 
            where the key states which EV follows that route
            
- DepartVector: is a tuple containing three floats which represent the state of a single EV when it departs from the 
                depot:
    [0] The time of the day the EV depart from the depot
    [1] The SOC when the EV departs from the depot
    [2] The payload when the EV departs from the depot
    
- DepartDict: is a dictionary containing the departing information for each EV. The key is the id of the EV assigned
                to the corresponding initial state, which is a DepartVector.
"""
RouteVector = Tuple[Tuple[int, ...], Tuple[float, ...], Tuple[float, ...]]  # (S, L, w1)
RouteDict = Dict[int, RouteVector]

DepartVector = Tuple[float, float, float]  # (x1_0, x2_0, x3_0)
DepartDict = Dict[int, DepartVector]

"""
Routes XML file read-out handlers
"""


def route_to_element(route: RouteVector) -> ET.Element:
    element = ET.Element('route')
    for (Sk, Lk, w1k) in zip(*route):
        attr = {'node': str(Sk), 'SOC_increment': str(Lk), 'wait_after_service': str(w1k)}
        _stop = ET.SubElement(element, 'stop', attrib=attr)
    return element


def route_from_element(route_element: ET.Element) -> Union[RouteVector]:
    if not route_element:
        return tuple(), tuple(), tuple()
    S = []
    L = []
    w1 = []
    for _stop in route_element:
        _stop: ET.Element
        Sk = int(_stop.get('node'))
        Lk = float(_stop.get('SOC_increment'))
        w1k = float(_stop.get('wait_after_service'))

        S.append(Sk)
        L.append(Lk)
        w1.append(w1k)
    return tuple(S), tuple(L), tuple(w1)


def write_routes(filepath: Path, routes: RouteDict, depart_info: DepartDict = None,
                 write_pretty: bool = False) -> ET.Element:
    """
    Saves specified routes to an XML file. This XML file will be used by the vehicles to operate. It overwrites the
    specified file.
    :param filepath: location of the XML file
    :param routes: collection of routes per vehicle
    :param depart_info: dictionary containing departing state for each EV (from depot). Default=None (ignored)
    :param write_pretty: write as pretty XML. Default: False
    :return: the ElementTree instance
    """
    tree_element = ET.Element('routes')
    for id_ev, route in routes.items():
        _ev = ET.SubElement(tree_element, 'vehicle', attrib={'id': str(id_ev)})
        if depart_info:
            try:
                x1_0, x2_0, x3_0 = depart_info[id_ev]
                dep_attrib = {'time': str(x1_0), 'soc': str(x2_0), 'payload': str(x3_0)}
                _depart = ET.SubElement(_ev, 'depart_info', attrib=dep_attrib)
            except KeyError:
                raise KeyError(f"No depart information was passed for EV {id_ev}")
        _ev.append(route_to_element(route))

    # Save
    if write_pretty:
        write_pretty_xml(filepath, tree_element)
    else:
        ET.ElementTree(tree_element).write(filepath)

    return tree_element


def read_routes(filepath: Path, read_depart_info: bool = False) -> Tuple[RouteDict, Union[DepartDict, None]]:
    """
    Reads a route's XML file and return a dictionary containing the routes (without initial conditions).
    :param filepath: the file file path
    :param read_depart_info: choose if reading departure information. Default: False
    :return: dictionary of routes
    """
    routes = {}
    depart_info = {} if read_depart_info else None
    tree_element = ET.parse(filepath).getroot()
    for _vehicle in tree_element:
        _vehicle: ET.Element
        id_ev = int(_vehicle.get('id'))
        route = route_from_element(_vehicle.find('route'))
        if route:
            routes[id_ev] = route
            if read_depart_info:
                _depart_info = _vehicle.find('depart_info')
                x1_0 = float(_depart_info.get('time'))
                x2_0 = float(_depart_info.get('soc'))
                x3_0 = float(_depart_info.get('payload'))
                depart_info[id_ev] = (x1_0, x2_0, x3_0)

    return routes, depart_info


def from_xml_element(element: ET.Element):
    t = element.get('type')
    cls = globals()[t]
    return cls.from_xml_element(element)
