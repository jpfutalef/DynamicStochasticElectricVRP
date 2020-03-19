from res.Fleet import Fleet
from res.Network import Network
import xml.etree.ElementTree as ET


class Parser:
    def __init__(self, path):
        self.path = path
        self.network = None
        self.fleet = None


class RawObserver:
    def __init__(self):
        pass


class EVRPObserver:
    def __init__(self):
        pass
