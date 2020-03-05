import numpy as np
import pandas as pd
import networkx as nx
from res.Network import Network
from res.Node import CustomerNode, ChargeStationNode, DepotNode


class ElectricVehicle:
    # TODO add documentation
    ev_id: int
    node_sequence: np.ndarray
    charging_sequence: np.ndarray
    depart_time: float
    network: Network

    def __init__(self, ev_id, network, max_payload=2.0, battery_capacity=200.0, max_tour_duration=300.0,
                 alpha_down=40.0, alpha_up=80.0, attrib=None):
        self.id = ev_id
        self.network = network

        self.alpha_up = alpha_up
        self.alpha_down = alpha_down
        self.max_tour_duration = max_tour_duration
        self.battery_capacity = battery_capacity
        self.max_payload = max_payload
        self.attrib = attrib

        # Initial conditions
        self.x1_0 = 0.0
        self.x2_0 = 0.0
        self.x3_0 = 0.0

        # Variables obtained by iterating state
        self.travel_time = {}
        self.energy_consumption = {}
        self.spent_time = {}
        self.x_reaching = []
        self.x_leaving = []
        self.iteration_done = False

        # Following variables are set after assignation problem is solved
        self.customers_to_visit = []
        self.ni = 0

    def set_sequences(self, node_sequence, charging_sequence, depart_time, x2_0):
        self.node_sequence = node_sequence
        self.charging_sequence = charging_sequence
        self.x1_0 = depart_time
        self.x2_0 = x2_0
        self.x3_0 = np.sum([self.network.nodes[i]['attr'].demand for i in node_sequence])

        self.travel_time = {}
        self.energy_consumption = {}
        self.spent_time = {}
        self.x_reaching = []
        self.x_leaving = []
        self.iteration_done = False

    def set_customers_to_visit(self, id_customers):
        self.customers_to_visit = id_customers
        self.ni = len(id_customers)

    def F1(self, x_prev: np.ndarray, node_from, node_to, soc_increment):
        # Calculate values
        travel_time = self.network.t(node_from, node_to, time_of_day=0.0)
        spent_time = self.network.spent_time(node_from, x_prev[1], soc_increment)
        x1_leaving_previous = x_prev[0] + spent_time
        x1_reaching_current = x1_leaving_previous + travel_time

        # Store travel times and spent times
        from_dict = self.travel_time[node_from] = {}
        from_dict[node_to] = travel_time

        self.spent_time[node_from] = spent_time

        # Return
        return x1_leaving_previous, x1_reaching_current

    def F2(self, x_prev: np.ndarray, node_from, node_to, soc_increment):
        # Calculate values
        energy_consumption = self.network.e(node_from, node_to, x_prev[2], time_of_day=0.0)
        x2_leaving_previous = x_prev[1] + soc_increment
        x2_reaching_current = x2_leaving_previous - energy_consumption

        # Store energy consumption values
        from_dict = self.energy_consumption[node_from] = {}
        from_dict[node_to] = energy_consumption

        # Return
        return x2_leaving_previous, x2_reaching_current

    def F3(self, x_prev: np.ndarray, node_from, node_to, soc_increment):
        x3_current = x_prev[2] - self.network.demand(node_from)
        return x3_current, x3_current

    def transition_function(self, x_prev, node_from, node_to, soc_increment):
        return np.array([self.F1(x_prev, node_from, node_to, soc_increment),
                         self.F2(x_prev, node_from, node_to, soc_increment),
                         self.F3(x_prev, node_from, node_to, soc_increment)])

    def iterateState(self):
        if self.iteration_done:
            return self.x_reaching, self.x_leaving
        else:
            x_prev = np.array([self.x1_0, self.x2_0, self.x3_0])
            x_leaving = np.zeros((3, len(self.node_sequence)))
            x_reaching = np.zeros((3, len(self.node_sequence)))

            node_from = 0
            charging_amount_from = 0

            for k, (node_to, charging_amount_to) in enumerate(zip(self.node_sequence, self.charging_sequence)):
                if k == 0:
                    x = np.concatenate((np.vstack(x_prev), np.vstack(x_prev)), axis=1)
                else:
                    x = self.transition_function(x_prev, node_from, node_to, charging_amount_from)
                    x_leaving[:, k - 1] = x[:, 0]
                x_reaching[:, k] = x[:, 1]
                x_prev = x[:, 1]
                node_from = node_to
                charging_amount_from = charging_amount_to
            x_leaving[:, -1] = x_reaching[:, -1]

            # Store the results of iteration
            self.iteration_done = True
            self.x_reaching = x_reaching
            self.x_leaving = x_leaving

            return x_reaching, x_leaving

    # Following methods should be used after running ~.iterateState() method
    def get_travel_times(self):
        return self.travel_time

    def get_energy_consumption(self):
        return self.energy_consumption

    def get_spent_times(self):
        return self.spent_time

    def cost_travel_time(self):
        tt = self.travel_time

    def cost_charging_time(self):
        st = self.spent_time

    def cost_energy_consumption(self):
        ec = self.energy_consumption

    def cost_charging_cost(self):
        ec = self.energy_consumption

    def reset(self):
        self.set_sequences(self.node_sequence, self.charging_sequence, self.x1_0, self.x2_0)


def createOptimizationVector(vehicles):
    """
    Creates optimization vector, assuming all EVs have already ran the ev.iterate() method.
    :param vehicles: dict with already-iterated vehicles by id
    :return: the optimization vector
    """
    # FIXME preallocate to arrays before to make more efficient

    S = []
    L = []
    X1 = []
    X2 = []
    X3 = []

    for id_vehicle, vehicle in vehicles.items():
        vehicle: ElectricVehicle

        S += vehicle.node_sequence
        L += vehicle.charging_sequence

        X1 += list(vehicle.x_reaching[0, :])
        X2 += list(vehicle.x_reaching[1, :])
        X3 += list(vehicle.x_reaching[2, :])

    V = np.vstack(S + L + X1 + X2 + X3)
    return V


def feasible(x, vehicles):
    """
    Checks feasibility of the optimization vector x. It's been assumed all EVs have already ran the
    ev.iterate() method.
    :param x: optimization vector
    :param vehicles: dict with vehicles info by id
    :return: Tuple (feasibility, distance) where feasibility=True and distance=0 if x is feasible; otherwise,
    feasibility=False and distance>0 if x isn't feasible. Distance is the squared accumulated distance.
    """
    is_feasible = True
    distance = 0

    # Amount of vehicles and customers, and the network information dictionary
    nVehicles = len(vehicles)
    nCustomers = np.sum([x.ni for _, x in vehicles.items()])
    network = vehicles[0].network

    networkSize = len(network.ids_customer) + len(network.ids_depot) + len(network.ids_charge_stations)

    # Amount of rows each constraint will occupy at the A matrix
    sumSi = np.sum([len(vehicle.node_sequence) for _, vehicle in vehicles.items()])

    # The following variable tracks the amount of rows per constraint
    constraintRows = 0

    # 16
    constraintRows += nVehicles

    # 17
    constraintRows += nCustomers

    # 18
    constraintRows += nCustomers

    # 25.1
    constraintRows += sumSi

    # 25.2
    constraintRows += sumSi

    # 26.1
    constraintRows += sumSi

    # 26.2
    constraintRows += sumSi

    # K0
    lenK0 = 2 * sumSi - 2 * nVehicles

    # Create linear inequalities matrix Ax <= b
    sizeOfOpVector = 5 * sumSi
    A = np.zeros((constraintRows, sizeOfOpVector))
    b = np.zeros((constraintRows, 1))

    # Start filling
    rowToChange = 0

    # 16
    i1 = 2 * sumSi
    i2 = 2 * sumSi
    for j, vehicle in vehicles.items():
        vehicle: ElectricVehicle
        i2 += len(vehicle.node_sequence) - 1

        A[rowToChange, i1] = -1.0
        A[rowToChange, i2] = 1.0
        b[rowToChange] = vehicle.max_tour_duration
        rowToChange += 1
        # print('16:', rowToChange)
        i1 += len(vehicle.node_sequence)
        i2 += 1

    # 17 & 18
    i1 = 2 * sumSi
    for j, vehicle in vehicles.items():
        vehicle: ElectricVehicle
        for nodeID in vehicle.node_sequence:
            if network.isCustomer(nodeID):
                A[rowToChange, i1] = -1.0
                b[rowToChange] = -networkDict[nodeID].timeWindowDown
                rowToChange += 1
                # print('17:', rowToChange)

                A[rowToChange, i1] = 1.0
                b[rowToChange] = networkDict[nodeID].timeWindowUp - networkDict[nodeID].serviceTime
                rowToChange += 1
                # print('18:', rowToChange)
            i1 += 1

    # 25.1 & 25.2
    i1 = 3 * sumSi
    for i, j in enumerate(vehiclesDict):
        for k in range(len(nodeSequences[j])):
            A[rowToChange, i1] = -1.0
            b[rowToChange] = -vehiclesDict[j].alphaDown
            rowToChange += 1
            # print('25.1:', rowToChange)

            A[rowToChange, i1] = 1.0
            b[rowToChange] = vehiclesDict[j].alphaUp
            rowToChange += 1
            # print('25.2:', rowToChange)
            i1 += 1

    # 26.1 & 26.2
    i1 = 3 * sumSi
    for i, j in enumerate(vehiclesDict):
        for k in range(len(nodeSequences[j])):
            A[rowToChange, i1] = -1.0
            b[rowToChange] = -vehiclesDict[j].alphaDown + vehiclesDict[j].chargingSequence[k]
            rowToChange += 1
            # print('25.1:', rowToChange)

            A[rowToChange, i1] = 1.0
            b[rowToChange] = vehiclesDict[j].alphaUp - vehiclesDict[j].chargingSequence[k]
            rowToChange += 1
            # print('25.2:', rowToChange)
            i1 += 1

    return is_feasible, distance


