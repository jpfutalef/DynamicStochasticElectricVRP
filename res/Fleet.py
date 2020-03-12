from res.ElectricVehicle import ElectricVehicle
from res.Network import Network
from res.Node import CustomerNode, ChargeStationNode, DepotNode, NetworkNode
import xml.etree.ElementTree as ET


class Fleet(dict):
    vehicle: ElectricVehicle

    def __init__(self):
        super().__init__()
        self.network = Network()
        self.xml_path = ''

    def from_xml(self, path, real_time=False):
        self.xml_path = path
        tree = self.network.from_xml(path)
        _fleet = tree.find('fleet')

        # Real time: just update what's needed
        if real_time:
            pass

        # Non real time: instance and set attributes
        else:
            for _ev in _fleet:
                pass

        # Update instances



