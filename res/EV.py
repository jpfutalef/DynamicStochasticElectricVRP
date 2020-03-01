import numpy as np
import pandas as pd
import networkx as nx
from res.Network import Network
from res.Node import CustomerNode, ChargeStationNode, DepotNode


class ElectricVehicle:  # TODO add documentation
    ev_id: int
    node_sequence: np.ndarray
    charging_sequence: np.ndarray
    depart_time_sequence: float
    network: Network

    def __init__(self, ev_id, node_sequence, charging_sequence, depart_time, network,
                 max_payload=2.0, battery_capacity=200.0, max_tour_duration=300.0, alpha_down=40.0, alpha_up=80.0,
                 x1_0=0.0, x2_0=40.0, x3_0=2.0, attrib=None):
        self.id = ev_id
        self.node_sequence = node_sequence
        self.charging_sequence = charging_sequence
        self.depart_time = depart_time
        self.network = network

        self.alpha_up = alpha_up
        self.alpha_down = alpha_down
        self.max_tour_duration = max_tour_duration
        self.battery_capacity = battery_capacity
        self.max_payload = max_payload
        self.attrib = attrib

        self.x1_0 = x1_0
        self.x2_0 = x2_0
        self.x3_0 = x3_0

    def F1(self, x_prev: np.ndarray, node_from, node_to, soc_increment):
        travel_time = self.network.travel_time(node_from, node_to, time_of_day=0.0)
        spent_time = self.network.spent_time(node_from, x_prev[1], soc_increment)
        return x_prev[0] + spent_time + travel_time

    def F2(self, node_from, node_to, soc_increment, eta):
        return

    def F3(self, idK0, idK1, Lk, eta):
        return

    def indexInSequence(self, nodeId):
        return

    def stateUpdate(self, idK0, idK1, Lk, eta):
        return

    def returnToInitialCondition(self):
        return

    def createStateSequenceStatic(self, sequenceEta):  # TODO docu
        return

    def createReachingLeavingStates(self, sequenceEta):  # TODO docu
        return

    def updateSequences(self, nodeSequence, chargingSequence, x1):
        return
