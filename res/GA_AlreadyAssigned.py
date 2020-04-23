from typing import Dict, List, Tuple, Union, NamedTuple
import models.routetypes

import numpy as np
from random import randint
from random import uniform
from random import sample

from models.ElectricVehicle import ElectricVehicle
from models.Fleet import Fleet, InitialCondition
from models.Network import Network, DynamicNetwork


# CLASSES


# TYPES
IndividualType = List
IndicesType = Dict[int, Tuple[int, int, int]]
StartingPointsType = Dict[int, InitialCondition]
RouteVector = Tuple[Tuple[int, ...], Tuple[float, ...]]
RouteDict = Dict[int, Tuple[RouteVector, float, float, float]]


# FUNCTIONS


def decode(individual: IndividualType, indices: IndicesType, starting_points: StartingPointsType,
           allowed_charging_operations=2) -> RouteDict:
    """
    Decodes an individual to the corresponding node sequence and charging sequence. S is the node sequence with the
    following structure  S = [S1, ..., Sm], where Si = [Si(0),..., Si(s_0-1)].  The L structure is the same as S
    structure.
    :param allowed_charging_operations: number of charging operations each vehicle is allowed to perform
    :param individual: The coded individual
    :param indices: a
    :param starting_points: dictionary with information about the start of sequences {id_vehicle:(S0, L0, x1_0,
    x2_0, x3_0),..., }
    :return: A 3-size tuple (S, L, x0)
    """
    # print("individual to decode: ", individual)
    routes = {}

    for id_ev, (i0, i1, i2) in indices.items():
        initial_condition = starting_points[id_ev]
        charging_operations = [(individual[i1 + 3 * i], individual[i1 + 3 * i + 1], individual[i1 + 3 * i + 2])
                               for i in range(allowed_charging_operations) if individual[i1 + 3 * i] != -1]
        customers_block: List[int] = individual[i0:i1]
        depart_time = individual[i2]
        ni = len(customers_block)

        init_node_seq = [initial_condition.S0] + customers_block
        init_charging_seq = [initial_condition.L0] + [0] * ni

        # case: there are not recharging operations
        if len(charging_operations) == 0:
            node_sequence = init_node_seq
            charging_sequence = init_charging_seq

        # case: there are recharging operations
        else:
            node_sequence = [0] * (ni + len(charging_operations) + 1)
            charging_sequence = [0.] * (ni + len(charging_operations) + 1)

            customers_after = [x[0] for x in charging_operations]

            iseq = 0
            iop = 0
            insert_cs = False
            for j, _ in enumerate(node_sequence):
                if insert_cs:
                    node_sequence[j] = charging_operations[iop][1]
                    charging_sequence[j] = charging_operations[iop][2]
                    insert_cs = False
                elif init_node_seq[iseq] in customers_after:
                    node_sequence[j] = init_node_seq[iseq]
                    charging_sequence[j] = init_charging_seq[iseq]
                    iop = customers_after.index(init_node_seq[iseq])
                    iseq += 1
                    insert_cs = True
                else:
                    node_sequence[j] = init_node_seq[iseq]
                    charging_sequence[j] = init_charging_seq[iseq]
                    iseq += 1

        # Store in dictionary
        S = tuple(node_sequence + [0])
        L = tuple(charging_sequence + [0])
        routes[id_ev] = ((S, L), depart_time, initial_condition.x2_0, initial_condition.x3_0)

    return routes


def mutate(individual: IndividualType, indices: IndicesType, starting_points: StartingPointsType,
           customers_to_visit: Dict[int, Tuple[int, ...]] , charging_stations: Tuple, allowed_charging_operations=2, index=None) -> None:
    # Choose a random index if not passed
    if index is None:
        index = randint(0, len(individual))

    # Find block
    for id_ev, (i0, i1, i2) in indices.items():
        if i0 <= index <= i2:
            # Case customer
            if i0 <= index < i1:
                i = randint(i0, i1 - 1)
                while True:
                    j = randint(i0, i1 - 1)
                    if j != i:
                        break
                swap_elements(individual, i, j)
                return

            # Case CS
            elif i1 <= index < i2:
                # Find operation sub-block
                for j in range(allowed_charging_operations):
                    if i1 + j * 3 <= index <= i1 + j * 3 + 2:
                        # Choose if making a charging operation
                        if randint(0, 1):
                            # Choose a customer node randomly
                            sample_space = (starting_points[id_ev].S0,) + customers_to_visit[id_ev]
                            while True:
                                customer = sample(sample_space, 1)[0]
                                # Ensure customer has not been already chosen
                                if customer not in [individual[i1 + 3 * x] for x in range(allowed_charging_operations)]:
                                    break
                            individual[i1 + 3 * j] = customer

                        else:
                            individual[i1 + 3 * j] = -1

                        # Choose a random CS anyways
                        individual[i1 + 3 * j + 1] = sample(charging_stations, 1)[0]

                        # Change amount anyways
                        amount = uniform(-30.0, 30.0)
                        individual[i1 + 3 * j + 2] += float(f"{amount:.2f}")
                        return

            # Case depart time
            else:
                amount = uniform(-60.0, 60.0)
                individual[i2] += amount
                return


def crossover(ind1: IndividualType, ind2: IndividualType, indices: IndicesType, allowed_charging_operations=2,
              index=None) -> None:
    # Choose a random index if not passed
    if index is None:
        index = randint(0, len(ind1))

    # Find block
    for id_ev, (i0, i1, i2) in indices.items():
        if i0 <= index <= i2:
            # Case customer
            if i0 <= index < i1:
                swap_block(ind1, ind2, i0, i1)
                return

            # Case CS
            elif i1 <= index < i2:
                swap_block(ind1, ind2, i1, i2)
                return

            # Case depart time
            else:
                swap_block(ind1, ind2, i2, i2)


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


def block_indices(customers_to_visit: Dict[int, Tuple[int, ...]], allowed_charging_operations=2) -> IndicesType:
    """
    Creates the indices of sub blocks.
    :param customers_to_visit:
    :param allowed_charging_operations:
    :return: tuples list with indices [(i0, j0), (i1, j1),...] where iN represent location of the beginning of customers_per_vehicle
    block and jN location of the beginning of charging operations block of evN in the individual, respectively.
    """
    indices = {}

    i0 = 0
    i1 = 0
    i2 = 0
    for id_vehicle, customers in customers_to_visit.items():
        ni = len(customers)
        i1 += ni
        i2 += ni + 3*allowed_charging_operations

        indices[id_vehicle] = (i0, i1, i2)

        i0 += ni + 3 * allowed_charging_operations + 1
        i1 += 3 * allowed_charging_operations + 1
        i2 += 1

    return indices


def random_individual(indices: IndicesType, starting_points: StartingPointsType,
                      customers_to_visit: Dict[int, Tuple[int, ...]], charging_stations: Tuple[int, ...],
                      allowed_charging_operations=2):
    """
    Creates a random individual
    :param allowed_charging_operations:
    :return: a random individual
    """
    individual = []
    for id_ev, (i0, i1, i2) in indices.items():
        init_point, customers = starting_points[id_ev], customers_to_visit[id_ev]
        customer_sequence = sample(customers, len(customers))
        sample_space = customers + (init_point.S0,) + (-1,) * allowed_charging_operations
        after_customers = sample(sample_space, allowed_charging_operations)
        charging_sequence = [0.] * allowed_charging_operations * 3
        for i, customer in enumerate(after_customers):
            charging_sequence[3 * i] = customer
            charging_sequence[3 * i + 1] = sample(charging_stations, 1)[0]
            amount = uniform(0.0, 90.0)
            charging_sequence[3 * i + 2] = float(f"{amount:.2f}")
        depart_time = [uniform(60*8, 60*18)]
        individual += customer_sequence + charging_sequence + depart_time
    return individual


def fitness(individual: IndividualType, fleet: Fleet, indices: IndicesType, starting_points:StartingPointsType,
            weights=(1.0, 1.0, 1.0, 1.0), penalization_constant=500000, allowed_charging_operations=2):
    """
    Calculates fitness of individual.
    :param indices:
    :param individual: The individual to decode
    :param fleet: dictionary with vehicle instances. They must have been assigned to customers_per_vehicle
    :param starting_points: dictionary with info of the initial state {id_vehicle:(S0, L0, x1_0, x2_0)}
    :param allowed_charging_operations: maximum charging operations per ev
    :param weights: tuple with weights of each variable in cost function (w1, w2, w3, w4)
    :param penalization_constant: positive number that represents the penalization of unfeasible individual
    :return: the fitness of the individual
    """

    # Decode
    routes = decode(individual, indices, starting_points, allowed_charging_operations=allowed_charging_operations)

    # Set routes
    fleet.set_routes_of_vehicles(routes)

    # Get optimization vector
    fleet.create_optimization_vector()

    # Cost
    cost_tt, cost_ec, cost_chg_op, cost_chg_cost = fleet.cost_function(weights[0], weights[1], weights[2], weights[3])

    # Check if the solution is feasible
    feasible, penalization = fleet.feasible()

    # penalization
    if not feasible:
        penalization += penalization_constant

    costs = np.array([cost_tt, cost_ec, cost_chg_op, cost_chg_cost])
    fit = np.dot(costs, np.asarray(weights)) + penalization

    return fit, feasible
