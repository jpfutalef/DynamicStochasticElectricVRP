from random import randint, uniform, sample, random
from typing import Dict, List, Tuple

import numpy as np

from Fleet import *
from GATools import *

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
    Positive fitness of the individual.
    :param individual:
    :param fleet:
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


def mutate(individual: IndividualType, m: int, num_customers: int, num_cs: int, r=2, index=None,
           block_probability=(.33, .33, .33)) -> IndividualType:
    """Mutates individual internally and returns it for optional assignation"""
    idt = len(individual) - m
    ics = idt - 3 * m * r

    # Choose a random index if not passed
    if index is None:
        # index = randint(0, len(individual) - 1)
        index = random_block_index(m, ics, idt, block_probability)

    # Case customer
    if 0 <= index < ics:
        case = random()
        if case < 0.4:
            mutate_customer1(individual, index, m, num_customers, num_cs, r)
        elif case < 0.8:
            mutate_customer2(individual, index, m, num_customers, num_cs, r)
        else:
            mutate_customer3(individual, index, m, num_customers, num_cs, r)

    # Case charging stations
    elif ics <= index < idt:
        case = random()
        if case < 0.8:
            mutate_charging_operation1(individual, index, m, num_customers, num_cs, r)
        else:
            mutate_charging_operation2(individual, index, m, num_customers, num_cs, r)

    # Case departure time
    else:
        mutate_departure_time1(individual, index, m, num_customers, num_cs, r)

    return individual


def crossover(ind1: IndividualType, ind2: IndividualType, m: int, r: int, index=None) -> Tuple[IndividualType,
                                                                                               IndividualType]:
    idt = len(ind1) - m
    ics = idt - 3 * m * r
    # Choose a random index if not passed
    if index is None:
        # index = randint(0, len(ind1))
        index = random_block_index(m, ics, idt, (.33, .33, .33))

    return crossover1(ind1, ind2, m, r, index)


def random_individual(num_customers, num_cs, m, r):
    customers = list(range(1, num_customers + 1))
    charging_stations = list(range(num_customers + 1, num_customers + num_cs + 1))

    # Customer and departure time blocks
    customer_blocks, departure_time_blocks = [], [0] * m
    for i in range(m):
        if i == m - 1:
            assigned_customers = sample(customers, len(customers))
        else:
            assigned_customers = sample(customers, randint(0, len(customers)))
        customer_blocks += assigned_customers + ['|']
        departure_time_blocks[i] = uniform(60 * 5, 60 * 12)
        customers = [i for i in customers if i not in assigned_customers]

    # Charging station blocks
    charging_operation_blocks = []
    for i in range(m * r):
        cust, cs, amount = sample(customers + [-1] * m * r, 1)[0], sample(charging_stations, 1)[0], uniform(20, 30)
        # cust, cs, amount = -1, sample(charging_stations, 1)[0], uniform(20, 30)
        charging_operation_blocks += [cust, cs, amount]

    individual = customer_blocks + charging_operation_blocks + departure_time_blocks
    return individual


# FUNCTIONS MAIN FUNCTIONS WILL CALL
def mutate_customer1(individual: IndividualType, index: int, m: int, num_customers: int, num_cs: int, r=2):
    """Swaps customers of a single vehicle"""
    index_start, index_end = customer_block_indices(individual, index)
    if index_end in (index_start, index_start + 1):
        return
    index_end -= 1
    i1, i2 = randint(index_start, index_end), randint(index_start, index_end)
    while i2 == i1:
        i2 = randint(index_start, index_end)
    swap_elements(individual, i1, i2)
    return individual


def mutate_customer2(individual: IndividualType, index: int, m: int, num_customers: int, num_cs: int, r=2):
    """Swaps two values in the customers range"""
    idt = len(individual) - m
    ics = idt - 3 * m * r
    if index >= ics - 1:
        return individual
    i1, i2 = index, randint(0, ics - 2)
    while i2 == i1:
        i2 = randint(0, ics - 2)
    swap_elements(individual, i1, i2)
    return individual


def mutate_customer3(individual: IndividualType, index: int, m: int, num_customers: int, num_cs: int, r=2):
    """Takes a customer from a vehicles and gives it to another vehicle"""
    idt = len(individual) - m
    ics = idt - 3 * m * r
    if index >= ics - 1:
        return individual
    i1, i2 = index, randint(0, ics - 2)
    while i2 == i1:
        i2 = randint(0, ics - 2)
    if i1 > i2:
        i1, i2 = i2, i1
    # Move i1 to i2
    c1 = individual[0:i1]
    c2 = individual[i1:i2]
    c3 = individual[i2 + 1:]
    val = [individual[i2]]
    individual = c1 + val + c2 + c3
    return individual


def mutate_charging_operation1(individual: IndividualType, index: int, m: int, num_customers: int, num_cs: int, r=2):
    """Mutates the charging operation block in the given index"""
    idt = len(individual) - m
    ics = idt - 3 * m * r

    offset = int((index - ics) / 3)
    op_index = ics + 3 * offset

    individual[op_index] = sample(range(1, num_customers + 1), 1)[0] if randint(0, 1) else -1
    individual[op_index + 1] = sample(range(num_customers + 1, num_customers + num_cs + 1), 1)[0]
    individual[op_index + 2] = abs(individual[op_index + 2] + uniform(-10, 10))


def mutate_charging_operation2(individual: IndividualType, index: int, m: int, num_customers: int, num_cs: int, r=2):
    """A new random charging operation"""
    customers = list(range(1, num_customers + 1))
    charging_stations = list(range(num_customers + 1, num_customers + num_cs + 1))

    idt = len(individual) - m
    ics = idt - 3 * m * r

    offset = int((index - ics) / 3)
    op_index = ics + 3 * offset

    cust, cs, amount = sample(customers + [-1] * m * r, 1)[0], sample(charging_stations, 1)[0], uniform(20, 30)
    individual[op_index] = cust
    individual[op_index + 1] = cs
    individual[op_index + 2] = amount


def mutate_departure_time1(individual: IndividualType, index: int, m: int, num_customers: int, num_cs: int, r=2):
    individual[index] += uniform(-60, 60)


def crossover1(ind1: IndividualType, ind2: IndividualType, m: int, r: int, index) -> Tuple[IndividualType,
                                                                                           IndividualType]:
    idt = len(ind1) - m
    ics = idt - 3 * m * r

    # Case customer
    if 0 <= index < ics:
        # Find customers blocks
        istart_1, iend_1 = customer_block_indices(ind1, index)
        istart_2, iend_2 = customer_block_indices(ind2, index)

        cust1 = ind1[istart_1:iend_1]
        cust2 = ind2[istart_2:iend_2]

        # Find common customers
        common = [i for i in cust1 for j in cust2 if i == j]

        offset = 0
        for i, customer in enumerate(ind1[istart_1: iend_1]):
            if customer not in common:
                sample_space1 = list(range(istart_1)) + list(range(iend_1, ics - 1))
                j = sample(sample_space1, 1)[0]
                move_from_to(ind1, i - offset + istart_1, j)
                iend_1 = iend_1 - 1 if j >= iend_1 else iend_1
                istart_1 = istart_1 + 1 if j < istart_1 else istart_1
                offset += 1

        offset = 0
        for i, customer in enumerate(ind2[istart_2: iend_2]):
            if customer not in common:
                sample_space2 = list(range(istart_2)) + list(range(iend_2, ics - 1))
                j = sample(sample_space2, 1)[0]
                move_from_to(ind2, i - offset + istart_2, j)
                iend_2 = iend_2 - 1 if j >= iend_2 else iend_2
                istart_2 = istart_2 + 1 if j < istart_2 else istart_2
                offset += 1

        # Swap blocks
        aux = ind1[istart_1:iend_1]
        ind1[istart_1:iend_1] = ind2[istart_2:iend_2]
        ind2[istart_2:iend_2] = aux

    # Case charging stations
    elif ics <= index < idt:
        offset = int((index - ics) / 3)
        op_index = ics + 3 * offset
        swap_block(ind1, ind2, op_index, op_index + 3)

    # Case departure time
    else:
        swap_block(ind1, ind2, index, index + 1)

    return ind1, ind2


# AUXILIARY FUNCTIONS
def swap_elements(l, i, j):
    l[i], l[j] = l[j], l[i]


def swap_block(l1, l2, i, j):
    l1[i: j], l2[i:j] = l2[i:j], l1[i:j]


def get_customers(individual, m):
    customer_sequences = []
    i0, i1 = 0, 0
    for i in range(m):
        i1 += individual[i0:].index('|')
        customer_sequences.append(individual[i0:i1])
        i0 = i1 + 1
        i1 = i0
    return customer_sequences


def customer_block_indices(individual, index):
    """
    First and last indices of the sub-block of customers based on index
    :param individual: the individual
    :param index: position in the individual
    :return: first and last indices of the sub-block of customers where the position is contained. If there are no
    customers, both indices will be the same.
    """
    index_end = index if (individual[index] == '|') else index + individual[index:].index('|')
    if not index_end:
        return 0, 0
    try:
        index_start = index_end - individual[index_end - 1::-1].index('|')
    except ValueError:
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


def random_block_index(m, ics, idt, block_probability):
    if random() < block_probability[0]:
        return randint(0, ics - 1)
    elif random() < block_probability[1]:
        return randint(ics, idt - 1)
    else:
        return randint(idt, idt + m - 1)


def optimal_route_assignation(fleet: Fleet, hp: HyperParameters):
    # TOOLBOX
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin, feasible=False)

    toolbox = base.Toolbox()

    toolbox.register("individual", random_individual, num_customers=len(fleet.network.customers),
                     num_cs=len(fleet.network.charging_stations), m=len(fleet.vehicles), r=hp.r)
    toolbox.register("evaluate", fitness, fleet=fleet, starting_points=hp.starting_points, weights=hp.weights,
                     penalization_constant=hp.penalization_constant, r=hp.r)
    toolbox.register("mate", crossover, m=len(fleet.vehicles), r=hp.r, index=None)
    toolbox.register("mutate", mutate, m=len(fleet.vehicles), num_customers=len(fleet.network.customers),
                     num_cs=len(fleet.network.charging_stations),
                     r=hp.r, index=None)
    toolbox.register("select", tools.selTournament, tournsize=hp.tournament_size)
    toolbox.register("select_worst", tools.selWorst)
    toolbox.register("decode", decode, m=len(fleet.vehicles), fleet=fleet, starting_points=hp.starting_points,
                     r=hp.r)

    # BEGIN ALGORITHM
    t_init = time.time()

    # Random population
    pop = [creator.Individual(toolbox.individual()) for i in range(hp.num_individuals)]

    # Evaluate the initial population and get fitness of each individual
    for ind in pop:
        fit, feasible = toolbox.evaluate(ind)
        ind.fitness.values = (fit,)
        ind.feasible = feasible

    print(f'  Evaluated {len(pop)} individuals')
    fits = [ind.fitness.values for ind in pop]
    bestOfAll = tools.selBest(pop, 1)[0]

    # These will save statistics
    X, Ymax, Ymin, Yavg, Ystd, YbestInd = [], [], [], [], [], []

    print("################  Start of evolution  ################")
    # Begin the evolution
    for g in range(hp.max_generations):
        # A new generation
        print(f"-- Generation {g}/{hp.max_generations} --")
        X.append(g)

        # Update block probabilities
        if g < 50:
            block_probabilities = (.33, .33, .33)
        elif g < 100:
            block_probabilities = (.2, .6, .2)
        elif g < 150:
            block_probabilities = (.6, .2, .2)
        elif g < 200:
            block_probabilities = (.33, .33, .33)
        elif g < 250:
            block_probabilities = (.2, .6, .2)
        elif g < 300:
            block_probabilities = (.6, .2, .33)
        else:
            block_probabilities = (.33, .33, .33)

        # Select the best individuals, if given
        if hp.keep_best:
            best_individuals = list(map(toolbox.clone, tools.selBest(pop, hp.keep_best)))

        # Select and clone the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Mutation
        for mutant in offspring:
            if random() < hp.MUTPB:
                toolbox.mutate(mutant, block_probability=block_probabilities)
                del mutant.fitness.values

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random() < hp.MUTPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Evaluate the individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            fit, feasible = toolbox.evaluate(ind)
            ind.fitness.values = (fit,)
            ind.feasible = feasible

        print(f'  Evaluated {len(invalid_ind)} individuals')

        # The population is entirely replaced by a sorted offspring
        pop[:] = offspring
        pop[:] = tools.selBest(pop, len(pop))

        # Insert best individuals from previous generation
        if hp.keep_best:
            pop[:] = best_individuals + pop[:-hp.keep_best]

        # Update best individual
        bestInd = tools.selBest(pop, 1)[0]
        if bestInd.fitness.wvalues[0] > bestOfAll.fitness.wvalues[0]:
            bestOfAll = bestInd

        # Real-time info
        print(f"Best individual  : {bestInd}\n Fitness: {bestInd.fitness.wvalues[0]} Feasible: {bestInd.feasible}")

        worstInd = tools.selWorst(pop, 1)[0]
        print(f"Worst individual : {worstInd}\n Fitness: {worstInd.fitness.wvalues[0]} Feasible: {worstInd.feasible}")

        print(
            f"Curr. best-of-all: {bestOfAll}\n Fitness: {bestOfAll.fitness.wvalues[0]} Feasible: {bestOfAll.feasible}")

        # Statistics
        fits = [sum(ind.fitness.wvalues) for ind in pop]
        mean = np.average(fits)
        std = np.std(fits)

        print(f"Max {max(fits)}")
        print(f"Min {min(fits)}")
        print(f"Avg {mean}")
        print(f"Std {std}")

        Ymax.append(-max(fits))
        Ymin.append(-min(fits))
        Yavg.append(mean)
        Ystd.append(std)

        print()

    t_end = time.time()
    print("################  End of (successful) evolution  ################")

    algo_time = t_end - t_init
    print('Algorithm time:', algo_time)

    routes = toolbox.decode(bestOfAll)
    return routes, bestOfAll, toolbox