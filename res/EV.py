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
                 x2_0=40.0, x3_0=2.0, attrib=None):
        self.id = ev_id
        self.node_sequence = node_sequence
        self.charging_sequence = charging_sequence
        self.network = network

        self.alpha_up = alpha_up
        self.alpha_down = alpha_down
        self.max_tour_duration = max_tour_duration
        self.battery_capacity = battery_capacity
        self.max_payload = max_payload
        self.attrib = attrib

        self.x1_0 = depart_time
        self.x2_0 = x2_0
        self.x3_0 = x3_0

    def F1(self, x_prev: np.ndarray, node_from, node_to, soc_increment):
        # TODO hacer if node is cs?
        travel_time = self.network.travel_time(node_from, node_to, time_of_day=0.0)
        spent_time = self.network.spent_time(node_from, x_prev[1], soc_increment)
        x1_leaving_previous = x_prev[0] + spent_time
        x1_reaching_current = x1_leaving_previous + travel_time
        return x1_leaving_previous, x1_reaching_current

    def F2(self, x_prev: np.ndarray, node_from, node_to, soc_increment):
        # TODO hacer if node is cs?
        energy_consumption = self.network.energy_consumption(node_from, node_to, x_prev[2], time_of_day=0.0)
        x2_leaving_previous = x_prev[1] + soc_increment
        x2_reaching_current = x2_leaving_previous - energy_consumption
        return x2_leaving_previous, x2_reaching_current

    def F3(self, x_prev: np.ndarray, node_from, node_to, soc_increment):
        # TODO hacer if node is cs?
        x3_current = x_prev[2] - self.network.demand(node_from)
        return x3_current, x3_current

    def transition_function(self, x_prev, node_from, node_to, soc_increment):
        return np.vstack([self.F1(x_prev, node_from, node_to, soc_increment),
                          self.F2(x_prev, node_from, node_to, soc_increment),
                          self.F3(x_prev, node_from, node_to, soc_increment)])

    def iterateState(self):
        x_prev = np.vstack([self.x1_0, self.x2_0, self.x3_0])
        x_leaving = np.zeros((3, len(self.node_sequence)))
        x_reaching = np.zeros((3, len(self.node_sequence)))
        x_leaving[:, 0] = x_prev
        x_reaching[:, 0] = x_prev

        node_from = 0
        charging_amount_from = 0

        for k, (node_to, charging_amount_to) in enumerate(zip(self.node_sequence, self.charging_sequence)):
            if k == 0:
                node_from = node_to
                charging_amount_from = charging_amount_to
            else:
                x = self.transition_function(x_prev, node_from, node_to, charging_amount_from)
                x_prev = x[:, 0]
                x_reaching[:, k] = x[:, 0]
                x_leaving[:, k] = x[:, 1]

        return x_reaching, x_leaving

    def indexInSequence(self, nodeId):
        return

    def returnToInitialCondition(self):
        return

    def createStateSequenceStatic(self, sequenceEta):  # TODO docu
        return

    def createReachingLeavingStates(self, sequenceEta):  # TODO docu
        return

    def updateSequences(self, node_sequence, charging_sequence, depart_time):
        return
