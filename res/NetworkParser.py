# XML tools
import xml.etree.ElementTree as ET

# EV and network libraries
import res
from res.ElectricVehicle import ElectricVehicle
from res.Node import DepotNode, CustomerNode, ChargeStationNode
from res.Network import Network


class NetworkParser:
    def __init__(self):
        self.file = 0
        self.network = 0
        self.vehicles = 0

    def parse_init(self, path):
        self.file = tree = ET.parse(path)
        _info = tree.find('info')
        _network = tree.find('network')
        _fleet = tree.find('fleet')

        # [START Node data]
        _nodes = _network.find('nodes')
        _edges = _network.find('edges')
        _technologies = _network.find('technologies')

        nodes = []
        for _node in _nodes:
            node_type = _node.get('type')
            id_node = int(_node.get('id'))
            pos = (float(_node.get('cx')), float(_node.get('cy')))
            if node_type == '0':
                node = DepotNode(id_node, pos=pos)

            elif node_type == '1':
                tw_upp = float(_node.get('tw_upp'))
                tw_low = float(_node.get('tw_low'))
                demand = float(_node.get('request'))
                spent_time = float(_node.get('spent_time'))
                node = CustomerNode(id_node, spent_time, demand, tw_upp, tw_low, pos=pos)

            elif node_type == '2':
                index_technology = int(_node.get('technology')) - 1
                _technology = _technologies[index_technology]
                charging_times = []
                battery_levels = []
                for _bp in _technology:
                    charging_times.append(float(_bp.get('charging_time')))
                    battery_levels.append(float(_bp.get('battery_level')))
                capacity = _node.get('capacity')
                node = ChargeStationNode(id_node, capacity, charging_times, battery_levels, pos=pos)

            nodes.append(node)

        networkSize = len(nodes)

        print('There are', networkSize, 'nodes in the network.')
        # [END Node data]

        # [START Edge data]
        id_nodes = [x.id for x in nodes]
        travel_time = {}
        energy_consumption = {}
        coordinates = {}

        for i, nodeFrom in enumerate(_edges):
            tt_dict = travel_time[i] = {}
            ec_dict = energy_consumption[i] = {}
            for j, nodeTo in enumerate(nodeFrom):
                tt_dict[j] = float(nodeTo.get('travel_time'))
                ec_dict[j] = float(nodeTo.get('energy_consumption'))
            coordinates[i] = nodes[i].pos

        # Show stored values
        print('NODES IDSs:\n', id_nodes)
        print('RESULTING TIME MATRIX:\n', travel_time)
        print('RESULTING ENERGY CONSUMPTION MATRIX:\n', energy_consumption)
        print('RESULTING NODES COORDINATES:\n', coordinates)
        # [END Edge data]

        # Instance Network
        self.network = Network()
        self.network.set_nodes(nodes)
        self.network.set_travel_time(travel_time)
        self.network.set_energy_consumption(energy_consumption)
