from random import randint, uniform, sample, random
from typing import Dict, List, Tuple

import numpy as np

from models.Fleet import Fleet, InitialCondition

# CLASSES


# TYPES
IndividualType = List
IndicesType = Dict[int, Tuple[int, int, int]]
StartingPointsType = Dict[int, InitialCondition]
RouteVector = Tuple[Tuple[int, ...], Tuple[float, ...]]
RouteDict = Dict[int, Tuple[RouteVector, float, float, float]]


# FUNCTIONS


def decode(individual: IndividualType, indices: IndicesType, init_state: StartingPointsType) -> RouteDict:
    """
    """
    routes = {}
    for id_ev, (i0, i1, i2) in indices.items():
        Sk = individual[i0:i1]
        Lk = [0] * len(Sk)
        chg_ops = individual[i1:i2]
        offset = 1
        for i, _ in enumerate(Sk):
            charging_station, amount = chg_ops[2 * i], chg_ops[2 * i + 1]
            if charging_station != -1:
                Sk = Sk[:i+offset] + [charging_station] + Sk[i+offset:]
                Lk = Lk[:i+offset] + [amount] + Lk[i+offset:]
                offset += 1
        Sk = tuple([0] + Sk + [0])
        Lk = tuple([0] + Lk + [0])
        x0 = individual[i2]
        routes[id_ev] = ((Sk, Lk), x0, init_state[id_ev].x2_0, init_state[id_ev].x3_0)
    return routes


def mutate(individual: IndividualType, indices: IndicesType, charging_stations: Tuple, repeat=1) -> None:
    for i in range(repeat):
        index = random_block_index(indices)
        for id_ev, (i0, i1, i2) in indices.items():
            if i0 <= index <= i2:
                # Case customer
                if i0 <= index < i1:
                    case = random()
                    if case < 0.55:
                        i, j = randint(i0, i1 - 1), randint(i0, i1 - 1)
                        while i == j:
                            j = randint(i0, i1 - 1)
                        swap_elements(individual, i, j)
                    elif case < 0.95:
                        i = randint(i0, i1 - 1)
                        individual[i0:i1] = individual[i:i1] + individual[i0:i]
                    else:
                        individual[i0:i1] = sample(individual[i0:i1], i1 - i0)

                # Case CS
                elif i1 <= index < i2:
                    i = i1 + 2 * int((index - i1) / 2)
                    sample_space = [-1]*len(charging_stations) + list(charging_stations)
                    individual[i] = sample(sample_space, 1)[0]
                    individual[i + 1] = abs(individual[i + 1] + uniform(-10, 10))

                # Case depart time
                else:
                    amount = uniform(-20.0, 20.0)
                    individual[i2] = abs(individual[i2] + amount)
                break


def crossover(ind1: IndividualType, ind2: IndividualType, indices: IndicesType, repeat=1) -> None:
    for i in range(repeat):
        index = random_block_index(indices)
        for id_ev, (i0, i1, i2) in indices.items():
            if i0 <= index <= i2:
                # Case customer
                if i0 <= index < i1:
                    swap_block(ind1, ind2, i0, i1)
                # Case CS
                elif i1 <= index < i2:
                    i = i1 + 2 * int((index - i1) / 2)
                    case = random()
                    if case < 0.5:
                        swap_block(ind1, ind2, i1, i2)
                    elif case < 0.75:
                        swap_block(ind1, ind2, i1, i)
                    else:
                        swap_block(ind1, ind2, i, i2)
                # Case depart time
                else:
                    swap_block(ind1, ind2, i2, i2)
                break


def swap_elements(l, i, j):
    """
    This function allows to swap two elements in a list, given two positions.
    :param l: the list
    :param i: first position
    :param j: second position
    """
    l[i], l[j] = l[j], l[i]


def swap_block(l1, l2, i, j):
    """
    This function allows to swap block within a list, given two positions.
    :param l1: first list
    :param l2: second list
    :param i: first position+

    :param j: second position
    """
    l1[i: j], l2[i:j] = l2[i:j], l1[i:j]


def random_block_index(indices: IndicesType) -> int:
    id_ev = sample(indices.keys(), 1)[0]
    i0, i1, i2 = indices[id_ev]
    if random() < 0.33:
        return randint(i0, i1 - 1)
    elif random() < 0.33:
        return sample(range(i1, i2, 2), 1)[0]
    else:
        return i2


def block_indices(customers_to_visit: Dict[int, Tuple[int, ...]]) -> IndicesType:
    """
    Creates the indices of sub blocks.
    :param customers_to_visit:
    :return: tuples list with indices [(i0, j0), (i1, j1),...] where iN represent location of the beginning of customers_per_vehicle
    block and jN location of the beginning of charging operations block of evN in the individual, respectively.
    """
    indices = {}
    i0, i1, i2 = 0, 0, 0
    for id_vehicle, customers in customers_to_visit.items():
        ni = len(customers)
        i1 += ni
        i2 = i1 + 2*ni
        indices[id_vehicle] = (i0, i1, i2)
        i0 = i2 + 1
        i1, i2 = i0, i0

    return indices


def random_individual(customers_per_vehicle: Dict[int, Tuple[int, ...]], charging_stations: Tuple[int, ...]):
    """
    """
    sample_space = list(charging_stations) + [-1]
    individual = []
    for id_ev, customers in customers_per_vehicle.items():
        customer_sequence = sample(customers, len(customers))
        charging_operations = []
        for _ in customer_sequence:
            #charging_operations.append(sample(sample_space, 1)[0])
            charging_operations.append(-1)
            charging_operations.append(uniform(5, 80))
        depart_time = [uniform(60 * 9, 60 * 12)]
        individual += customer_sequence + charging_operations + depart_time
    return individual


def fitness(individual: IndividualType, fleet: Fleet, indices: IndicesType, init_state: StartingPointsType,
            weights=(1.0, 1.0, 1.0, 1.0), penalization_constant=500000):
    """
    Calculates fitness of individual.
    :param indices:
    :param individual: The individual to decode
    :param fleet: dictionary with vehicle instances. They must have been assigned to customers_per_vehicle
    :param init_state: dictionary with info of the initial state {id_vehicle:(S0, L0, x1_0, x2_0, x3_0)}
    :param weights: tuple with weights of each variable in cost function (w1, w2, w3, w4)
    :param penalization_constant: positive number that represents the penalization of unfeasible individual
    :return: the fitness of the individual
    """

    # Decode
    routes = decode(individual, indices, init_state)

    # Set routes
    fleet.set_routes_of_vehicles(routes)

    # Get optimization vector
    fleet.create_optimization_vector()

    # Cost
    cost_tt, cost_ec, cost_chg_op, cost_chg_cost = fleet.cost_function()

    # Check if the solution is feasible
    feasible, penalization = fleet.feasible()

    # penalization
    if not feasible:
        penalization += penalization_constant

    costs = np.array([cost_tt, cost_ec, cost_chg_op, cost_chg_cost])
    fit = np.dot(costs, np.asarray(weights)) + penalization

    return fit, feasible
