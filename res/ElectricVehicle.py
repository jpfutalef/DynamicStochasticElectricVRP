import numpy as np
import pandas as pd
import networkx as nx
from res.Network import Network
from res.Node import CustomerNode, ChargeStationNode, DepotNode


def cost_arc(item):
    cost = 0
    for node_from, d in item.items():
        for node_to, c in d.items():
            cost += c
    return cost


def cost_node(item):
    cost = 0
    for node, c in item:
        cost += c
    return cost


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
        self.spent_time = []
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
        self.x3_0 = np.sum([self.network.nodes[i]['attr'].demand for i in node_sequence if self.network.isCustomer(i)])

        self.travel_time = {}
        self.energy_consumption = {}
        self.spent_time = []
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

        self.spent_time.append(spent_time)

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
        return cost_arc(self.travel_time)

    def cost_charging_time(self):
        cost = 0
        for k, node in enumerate(self.node_sequence):
            if self.network.isChargeStation(node):
                cost += self.spent_time[k]
        return cost

    def cost_energy_consumption(self):
        return cost_arc(self.energy_consumption)

    def cost_charging_cost(self):
        cost = 0
        for k, node in enumerate(self.node_sequence):
            if self.network.isChargeStation(node):
                cost += self.charging_sequence[k]
        return cost

    def reset(self):
        self.set_sequences(self.node_sequence, self.charging_sequence, self.x1_0, self.x2_0)


def createOptimizationVector(vehicles):
    """
    Creates optimization vector, assuming all EVs have already ran the ev.iterate() method.
    :param vehicles: dict with already-iterated vehicles by id
    :return: the optimization vector
    """

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


def feasible(x: np.ndarray, vehicles: dict):
    """
    Checks feasibility of the optimization vector x. It's been assumed all EVs have already ran the
    ev.iterate() method.
    :param x: optimization vector
    :param vehicles: dict with vehicles info by id
    :return: Tuple (feasibility, distance) where feasibility=True and distance=0 if x is feasible; otherwise,
    feasibility=False and distance>0 if x isn't feasible. Distance is the squared accumulated distance.
    """
    vehicle: ElectricVehicle

    # Variables to return
    is_feasible = True
    dist = 0

    # Variables from the optimization vector and vehicles
    n_vehicles = len(vehicles)
    n_customers = np.sum([vehicle.ni for _, vehicle in vehicles.items()])
    network = vehicles[0].network
    sum_si = np.sum(len(vehicle.node_sequence) for _, vehicle in vehicles.items())
    lenght_op_vector = len(x)

    i_S, i_L, i_x1, i_x2, i_x3, i_theta = 0, sum_si, 2 * sum_si, 3 * sum_si, 4 * sum_si, 5 * sum_si

    # Amount of rows
    rows = 0

    rows += n_vehicles   # 2.16
    rows += n_customers  # 2.17
    rows += n_customers  # 2.18
    rows += sum_si  # 2.25-1
    rows += sum_si  # 2.25-2
    rows += sum_si  # 2.26-1
    rows += sum_si  # 2.26-2

    # Matrices
    A = np.zeros((rows, lenght_op_vector))
    b = np.zeros((rows, 1))

    # Start filling
    row = 0

    # 2.16
    si = 0
    for j, vehicle in vehicles.items():
        A[row, i_x1+si] = -1.0
        si += len(vehicle.node_sequence)
        A[row, i_x1+si-1] = 1.0
        b[row] = vehicle.max_tour_duration
        row += 1

    # 2.17 & 2.18
    si = 0
    for _, vehicle in vehicles.items():
        for k, Sk in enumerate(vehicle.node_sequence):
            if network.isCustomer(Sk):
                A[row, i_x1+si+k] = -1.0
                b[row] = -network.nodes[Sk]['attr'].timeWindowDown
                row += 1

                A[row, i_x1+si+k] = 1.0
                b[row] = network.nodes[Sk]['attr'].timeWindowUp - network.spent_time(Sk)
                row += 1
        si += len(vehicle.node_sequence)

    # 2.25-1 & 2.25-2
    si = 0
    for _, vehicle in vehicles.items():
        for k, Sk in enumerate(vehicle.node_sequence):
            A[row, i_x2+si+k] = -1.0
            b[row] = -vehicle.alpha_down
            row += 1

            A[row, i_x2+si+k] = 1.0
            b[row] = vehicle.alpha_up
            row += 1
        si += len(vehicle.node_sequence)

    # 2.26-1 & 2.26-2
    si = 0
    for _, vehicle in vehicles.items():
        for k, Sk in enumerate(vehicle.node_sequence):
            A[row, i_x2 + si + k] = -1.0
            b[row] = -vehicle.alpha_down + vehicle.charging_sequence[k]
            row += 1

            A[row, i_x2 + si + k] = 1.0
            b[row] = vehicle.alpha_up - vehicle.charging_sequence[k]
            row += 1
        si += len(vehicle.node_sequence)

    # Check
    mult = np.matmul(A, x)
    boolList = mult <= b
    for result in boolList:
        if not result:
            dist = distance(boolList, mult, b, vehicles)
            is_feasible = False
            break

    return is_feasible, dist


def distance(results, mult, b, vehicles):
    def dist_fun(x, y):
        # return np.abs(x - y)
        # return np.sqrt(np.power(x - y, 2))
        # return np.abs(np.power(x - y, 2))
        return np.power(x - y, 2)

    return np.sum([dist_fun(mult[i, 0], b[i, 0]) for i, result in enumerate(results) if result])


