from typing import Dict, List, Tuple, Union, NamedTuple
import models.routetypes

import numpy as np
from random import randint, uniform, sample, random

from models.ElectricVehicle import ElectricVehicle
from models.Fleet import Fleet, InitialCondition
from models.Network import Network

# TYPES
IndividualType = List
IndicesType = Dict[int, Tuple[int, int]]
StartingPointsType = Dict[int, InitialCondition]
RouteVector = Tuple[Tuple[int, ...], Tuple[float, ...]]
RouteDict = Dict[int, Tuple[RouteVector, float, float, float]]


# MAIN FUNCTIONS
def decode(individual: IndividualType, m: int, fleet: Fleet, starting_points: StartingPointsType, r=2) -> RouteDict:
    """
    Decodes individual to node sequence, charging sequence and initial conditions.
    :param individual: the individual
    :param m: fleet size
    :param fleet: fleet instance
    :param starting_points: dictionary with initial conditions of each EV
    :param r: maximum charging operations allowed per vehicle
    :return: the routes encoded by the individual
    """
    idt = len(individual) - m
    ics = idt - 3 * m * r

    customer_sequences = get_customers(individual, m)
    charging_sequences = [[0] * len(x) for x in customer_sequences]

    charging_operations = [(individual[ics + 3 * i], individual[ics + 3 * i + 1], individual[ics + 3 * i + 2])
                           for i in range(r * m)]
    depart_times = individual[idt:]

    # Insert charging operations
    for customer, chg_st, amount in charging_operations:
        if customer == -1:
            continue
        for node_sequence, charging_sequence in zip(customer_sequences, charging_sequences):
            if customer not in node_sequence:
                continue
            index = node_sequence.index(customer) + 1
            node_sequence.insert(index, chg_st)
            charging_sequence.insert(index, amount)

    # Store routes in dictionary
    routes = {}
    for id_ev, (node_sequence, charging_sequence, depart_time) in enumerate(zip(customer_sequences, charging_sequences,
                                                                                depart_times)):
        ic = starting_points[id_ev]
        S = tuple([ic.S0] + node_sequence + [0])
        L = tuple([ic.L0] + charging_sequence + [0])
        routes[id_ev] = ((S, L), depart_time, ic.x2_0, sum([fleet.network.demand(x) for x in S]))
    return routes


def fitness(individual: IndividualType, fleet: Fleet, starting_points: StartingPointsType,
            weights=(1.0, 1.0, 1.0, 1.0), penalization_constant=500000, r=2):
    """
    Fitness of given individual.
    :param individual:
    :param fleet:
    :param indices:
    :param starting_points:
    :param weights:
    :param penalization_constant:
    :param r:
    :return:
    """

    # Decode
    m = len(fleet.vehicles)
    routes = decode(individual, m, fleet, starting_points, r)

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


def mutate(individual: IndividualType, m: int, num_customers: int, num_cs: int, r=2,
           index=None) -> IndividualType:
    """
    Mutates individual internally and returns it for optional assignation.
    :param individual:
    :param m:
    :param num_customers:
    :param num_cs:
    :param r:
    :param index:
    :return:
    """
    # Indices separating customers, charging operations and departure times
    idt = len(individual) - m
    ics = idt - 3 * m * r

    # Choose a random index if not passed
    if index is None:
        index = randint(0, len(individual)-1)

    # Case customer
    if 0 <= index < ics:
        case = random()
        if case < 0.5:
            mutate_customer2(individual, index, m, r)
        else:
            mutate_customer3(individual, index, m, r)

    # Case charging stations
    elif ics <= index < idt:
        mutate_charging_operation1(individual, index, m, r, num_cs)

    # Case departure time
    else:
        mutate_departure_time1(individual, index, m, r)

    return individual


def crossover(ind1: IndividualType, ind2: IndividualType, m: int, r: int, index=None) -> Tuple[IndividualType,
                                                                                               IndividualType]:
    # Choose a random index if not passed
    if index is None:
        index = randint(0, len(ind1))

    return crossover1(ind1, ind2, m, r, index)


def random_individual(num_customers, num_cs, m, r):
    """
    Creates a random individual
    :param num_customers:
    :param num_cs:
    :param m:
    :param r:
    :return:
    """
    customers = list(range(1, num_customers + 1))
    charging_stations = list(range(num_customers + 1, num_customers + num_cs + 1))

    # Charging station blocks
    charging_operation_blocks = []
    for i in range(m * r):
        #cust = sample(customers + [-1], 1)[0]
        cust = -1
        cs = sample(charging_stations, 1)[0]
        amount = uniform(20, 30)
        charging_operation_blocks.append(cust)
        charging_operation_blocks.append(cs)
        charging_operation_blocks.append(amount)

    # Customer and departure time blocks
    bar_indices = sample(range(num_customers + m - 1), m - 1)
    customer_blocks = ['|' if x in bar_indices else 0 for x in range(num_customers + m)]
    customer_blocks[-1] = '|'

    departure_time_blocks = [0] * m

    index = 0
    for i in range(m):
        i0, i1 = customer_block_indices(customer_blocks, index)
        assigned_customers = [customers.pop(randint(0, len(customers) - 1)) for _ in range(i1 - i0 + 1)]
        customer_blocks[i0:i1 + 1] = assigned_customers

        departure_time_blocks[i] = uniform(60 * 5, 60 * 12)
        index = i1 + 2

    individual = customer_blocks + charging_operation_blocks + departure_time_blocks

    return individual


# FUNCTIONS MAIN FUNCTIONS WILL CALL
def mutate_customer1(individual: List, index: int, m: int, r: int):
    """Swaps customers of a single vehicle"""
    # If the current and previous values are 0, then the vehicle has no customers assigned
    if not (individual[index - 1] == '|' or individual[index] == '|'):
        return

    # Find vehicle operation
    index_start, index_end = customer_block_indices(individual, index)
    index_end += 1
    if index_end in [index_start, index_start+1]:
        return
    # Randomly choose two indices to swap
    i1 = randint(index_start, index_end)
    i2 = randint(index_start, index_end)
    while i2 == i1:
        i2 = randint(index_start, index_end)

    # Do swap
    swap_elements(individual, i1, i2)


def mutate_customer2(individual: List, index: int, m: int, r: int):
    """
    Swaps two values in the customers range
    :param individual:
    :param index:
    :param m:
    :param r:
    :return:
    """
    idt = len(individual) - m
    ics = idt - 3 * m * r

    if index >= ics - 1:
        return

    i1 = index
    i2 = randint(0, ics - 2)
    while i2 == i1:
        i2 = randint(0, ics - 2)

    # Do swap
    swap_elements(individual, i1, i2)


def mutate_customer3(individual: List, index: int, m: int, r: int):
    """
    Takes a customer from a vehicles and gives it to another vehicle
    :param individual:
    :param index:
    :param m:
    :param r:
    :return:
    """
    idt = len(individual) - m
    ics = idt - 3 * m * r

    if index >= ics - 1:
        return

    i1 = index
    i2 = randint(0, ics - 2)
    while i2 == i1:
        i2 = randint(0, ics - 2)

    if i1 > i2:
        i1, i2 = i2, i1

    # Move i1 to i2
    c1 = individual[0:i1]
    c2 = individual[i1:i2]
    c3 = individual[i2+1:]
    val = [individual[i2]]
    individual = c1 + val + c2 + c3


def mutate_charging_operation1(individual: List, index: int, m: int, r: int, num_cs: int):
    """
    Mutates the charging operation block in the given index
    :param individual:
    :param index:
    :param m:
    :param r:
    :param num_cs:
    :return:
    """
    idt = len(individual) - m
    ics = idt - 3 * m * r

    offset = int((index - ics) / 3)
    op_index = ics + 3 * offset

    individual[op_index] = sample(range(1, ics - m + 1), 1)[0] if randint(0, 1) else -1
    individual[op_index + 1] = sample(range(ics - m + 1, ics - m + 1 + num_cs), 1)[0]
    individual[op_index + 2] = abs(individual[op_index + 2] + uniform(-10, 10))


def mutate_departure_time1(individual: List, index: int, m: int, r: int):
    individual[index] += uniform(-100, 100)


def crossover1(ind1: IndividualType, ind2: IndividualType, m: int, r: int, index) -> Tuple[IndividualType,
                                                                                                 IndividualType]:
    idt = len(ind1) - m
    ics = idt - 3 * m * r

    # Case customer
    if 0 <= index < ics:
        # Find customers blocks
        istart_1, iend_1 = customer_block_indices(ind1, index)
        istart_2, iend_2 = customer_block_indices(ind2, index)

        cust1 = ind1[istart_1:iend_1 + 1]
        cust2 = ind2[istart_2:iend_2 + 1]

        # Find common customers
        common = [i for i in cust1 for j in cust2 if i == j]

        offset = 0
        for i, customer in enumerate(ind1[istart_1: iend_1 + 1]):
            if customer not in common:
                sample_space1 = list(range(istart_1)) + list(range(iend_1 + 1, ics - 1))
                j = sample(sample_space1, 1)[0]
                move_from_to(ind1, i - offset + istart_1, j)
                iend_1 = iend_1 - 1 if j > iend_1 else iend_1
                istart_1 = istart_1 + 1 if j < istart_1 else istart_1
                offset += 1

        offset = 0
        for i, customer in enumerate(ind2[istart_2: iend_2 + 1]):
            if customer not in common:
                sample_space2 = list(range(istart_2)) + list(range(iend_2 + 1, ics - 1))
                j = sample(sample_space2, 1)[0]
                move_from_to(ind2, i - offset + istart_2, j)
                iend_2 = iend_2 - 1 if j > iend_2 else iend_2
                istart_2 = istart_2 + 1 if j < istart_2 else istart_2
                offset += 1

        # Swap blocks
        aux = ind1[istart_1:iend_1 + 1]
        ind1[istart_1:iend_1 + 1] = ind2[istart_2:iend_2 + 1]
        ind2[istart_2:iend_2 + 1] = aux

    # Case charging stations
    elif ics <= index < idt:
        offset = int((index - ics) / 3)
        op_index = ics + 3 * offset
        swap_block(ind1, ind2, op_index, op_index + 3)

        # Case departure time
    else:
        swap_block(ind1, ind2, index, index + 1)

    return ind1, ind2

# AUXILIAR FUNCTIONS

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


def block_indices(customers_to_visit: Dict[int, Tuple[int, ...]], allowed_charging_operations=2):
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
    for id_vehicle, customers in customers_to_visit.items():
        ni = len(customers)
        i1 += ni

        indices[id_vehicle] = (i0, i1)

        i0 += ni + 3 * allowed_charging_operations
        i1 += 3 * allowed_charging_operations

    return indices

def get_customers(individual, m):
    customer_sequences = []
    i0 = 0
    i1 = 0
    for i in range(m):
        i1 += individual[i0:].index('|')
        customer_sequences.append(individual[i0:i1])
        i0 = i1 + 1
        i1 = i0
    return customer_sequences


def customer_block_indices(individual, index):
    # Find vehicle operation
    index_end = index - 1 if (individual[index] == '|') else index + individual[index:].index('|') - 1
    if index:
        try:
            index_start = index_end - individual[index_end::-1].index('|') + 1
        except ValueError:
            index_start = 0
    else:
        index_start = 0
    return index_start, index_end


def move_from_to(l, i, j):
    """
    Moves an item in a list by index
    :param l: the list where the item is
    :param i: old item index
    :param j: new item index
    :return: list with modifications
    """
    l.insert(j, l.pop(i))
    return l


def all_in_block(source, to_check):
    for i in to_check:
        if i not in source:
            return False
    return True