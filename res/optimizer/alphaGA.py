import os
import time
from random import randint, uniform, sample, random
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from deap import tools, base, creator

from optimizer.GATools import HyperParameters, GenerationsData, Fleet, RouteDict, IndividualType

'''
MAIN FUNCTIONS
'''


def decode(individual: IndividualType, fleet: Fleet, hp: HyperParameters) -> RouteDict:
    """
    Decodes individual to node sequence, charging sequence and initial conditions.
    :param individual: the individual
    :param fleet: fleet instance
    :param hp: GA hyper-parameters
    :return: the routes encoded by the individual
    """
    m = len(fleet)
    n = len(fleet.network.customers)

    ics = n + m
    idt = ics + 3 * m * hp.r

    customer_sequences = get_customers(individual, m)
    charging_sequences = [[0] * len(x) for x in customer_sequences]

    charging_operations = [(individual[ics + 3 * i], individual[ics + 3 * i + 1], individual[ics + 3 * i + 2])
                           for i in range(hp.r * m)]
    depart_times = individual[idt:]

    # Insert charging operations
    for customer, charging_station, amount in charging_operations:
        if customer == -1:
            continue
        for S, L in zip(customer_sequences, charging_sequences):
            try:
                index = S.index(customer) + 1
                if index == len(S):
                    S.insert(index, charging_station)
                    L.insert(index, amount)
                elif S[index] == charging_station:
                    L[index] += amount
                else:
                    S.insert(index, charging_station)
                    L.insert(index, amount)
            except ValueError:
                continue

    # Store routes in dictionary
    routes = {}
    for i, (S, L, dt) in enumerate(zip(customer_sequences, charging_sequences, depart_times)):
        S = tuple([0] + S + [0])
        L = tuple([0.] + L + [0.])
        routes[i] = ((S, L), dt, hp.alpha_up, sum([fleet.network.demand(x) for x in S]))
    return routes


def fitness(individual: IndividualType, fleet: Fleet, hp: HyperParameters):
    """
    Positive fitness of the individual.
    :param individual: the individual to evaluate
    :param fleet: fleet instance
    :param hp: GA hyper-parameters
    :return: fitness value of the individual (positive)
    """

    # Decode
    m = len(fleet.vehicles)
    routes = decode(individual, fleet, hp)

    # Set routes
    fleet.set_routes_of_vehicles(routes)

    # Cost
    costs = np.array(fleet.cost_function())

    # Calculate penalization
    feasible, distance, accept = fleet.feasible()

    penalization = 0
    if not accept:
        penalization = distance + hp.K1 + hp.K2
    elif accept and not feasible:
        penalization = distance + hp.K1

    # Calculate fitness
    fit = np.dot(costs, np.asarray(hp.weights)) + penalization

    return fit, feasible, accept


def mutate(individual: IndividualType, fleet: Fleet, hp: HyperParameters, block_probability=(.33, .33, .33),
           index=None) -> IndividualType:
    """Mutates individual internally and returns it for optional assignation"""
    m = len(fleet)
    r = hp.r
    num_customers = len(fleet.network.customers)
    num_cs = len(fleet.network.charging_stations)

    idt = len(individual) - m
    ics = idt - 3 * m * r

    # Choose a random index if not passed
    if index is None:
        index = random_block_index(m, ics, idt, block_probability)

    # [CASE 1] customer
    if 0 <= index < ics:
        case = random()
        if case < 0.4:
            mutate_customer1(individual, index, m, num_customers, num_cs, r)
        elif case < 0.8:
            mutate_customer2(individual, index, m, num_customers, num_cs, r)
        else:
            mutate_customer3(individual, index, m, num_customers, num_cs, r)

    # [CASE 2] Case charging stations
    elif ics <= index < idt:
        case = random()
        if case < 0.8:
            mutate_charging_operation1(individual, index, m, num_customers, num_cs, r)
        else:
            mutate_charging_operation2(individual, index, m, num_customers, num_cs, r)

    # [CASE 3] Departure time
    else:
        mutate_departure_time1(individual, index, m, num_customers, num_cs, r)

    return individual


def crossover(ind1: IndividualType, ind2: IndividualType, fleet: Fleet, hp: HyperParameters,
              block_probability=(.33, .33, .33), index=None) -> Tuple[IndividualType, IndividualType]:
    m = len(fleet)
    r = hp.r

    idt = len(ind1) - m
    ics = idt - 3 * m * r

    # Choose a random index if not passed
    if index is None:
        index = random_block_index(m, ics, idt, block_probability)

    # [CASE 1] Customer
    if 0 <= index < ics:
        # Find customers blocks
        i_start_1, i_end_1 = customer_block_indices(ind1, index)
        i_start_2, i_end_2 = customer_block_indices(ind2, index)

        customers_1 = ind1[i_start_1:i_end_1]
        customers_2 = ind2[i_start_2:i_end_2]

        # Find common customers
        common = [i for i in customers_1 for j in customers_2 if i == j]

        offset = 0
        for i, customer in enumerate(ind1[i_start_1: i_end_1]):
            if customer not in common:
                sample_space1 = list(range(i_start_1)) + list(range(i_end_1, ics - 1))
                j = sample(sample_space1, 1)[0]
                move_from_to(ind1, i - offset + i_start_1, j)
                i_end_1 = i_end_1 - 1 if j >= i_end_1 else i_end_1
                i_start_1 = i_start_1 + 1 if j < i_start_1 else i_start_1
                offset += 1

        offset = 0
        for i, customer in enumerate(ind2[i_start_2: i_end_2]):
            if customer not in common:
                sample_space2 = list(range(i_start_2)) + list(range(i_end_2, ics - 1))
                j = sample(sample_space2, 1)[0]
                move_from_to(ind2, i - offset + i_start_2, j)
                i_end_2 = i_end_2 - 1 if j >= i_end_2 else i_end_2
                i_start_2 = i_start_2 + 1 if j < i_start_2 else i_start_2
                offset += 1

        # Swap blocks
        aux = ind1[i_start_1:i_end_1]
        ind1[i_start_1:i_end_1] = ind2[i_start_2:i_end_2]
        ind2[i_start_2:i_end_2] = aux

    # [CASE 2] Case charging stations
    elif ics <= index < idt:
        offset = int((index - ics) / 3)
        op_index = ics + 3 * offset
        swap_block(ind1, ind2, op_index, op_index + 3)

    # [CASE 3] Departure time
    else:
        swap_block(ind1, ind2, index, index + 1)

    return ind1, ind2


'''
MUTATION OPERATIONS
'''


def mutate_customer1(individual: IndividualType, index: int, m: int, num_customers: int, num_cs: int, r: int):
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


def mutate_customer2(individual: IndividualType, index: int, m: int, num_customers: int, num_cs: int, r: int):
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


def mutate_customer3(individual: IndividualType, index: int, m: int, num_customers: int, num_cs: int, r: int):
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


def mutate_charging_operation1(individual: IndividualType, index: int, m: int, num_customers: int, num_cs: int, r: int):
    """Mutates the charging operation block in the given index"""
    idt = len(individual) - m
    ics = idt - 3 * m * r

    offset = int((index - ics) / 3)
    op_index = ics + 3 * offset

    a0 = list(range(1, num_customers + 1)) + [-1]*num_customers
    a1 = range(num_customers + 1, num_customers + num_cs + 1)
    individual[op_index] = sample(a0, 1)[0] if randint(0, 1) else individual[op_index]
    individual[op_index + 1] = sample(a1, 1)[0] if randint(0, 1) else individual[op_index + 1]
    # individual[op_index + 2] = abs(individual[op_index + 2] + uniform(-10, 10))
    individual[op_index + 2] = abs(individual[op_index + 2] + np.random.normal(0, 5))


def mutate_charging_operation2(individual: IndividualType, index: int, m: int, num_customers: int, num_cs: int, r: int):
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


def mutate_departure_time1(individual: IndividualType, index: int, m: int, num_customers: int, num_cs: int, r: int):
    # individual[index] += uniform(-60, 60)
    individual[index] += np.random.normal(0, 60)


'''
CROSSOVER FUNCTIONS
'''


def crossover_customer1(ind1: IndividualType, ind2: IndividualType, index: int, m: int, num_customers: int, num_cs: int,
                        r: int):
    idt = len(ind1) - m
    ics = idt - 3 * m * r

    # Find customers blocks
    i_start_1, i_end_1 = customer_block_indices(ind1, index)
    i_start_2, i_end_2 = customer_block_indices(ind2, index)

    customers_1 = ind1[i_start_1:i_end_1]
    customers_2 = ind2[i_start_2:i_end_2]

    # Find common customers
    common = [i for i in customers_1 for j in customers_2 if i == j]

    offset = 0
    for i, customer in enumerate(ind1[i_start_1: i_end_1]):
        if customer not in common:
            sample_space1 = list(range(i_start_1)) + list(range(i_end_1, ics - 1))
            j = sample(sample_space1, 1)[0]
            move_from_to(ind1, i - offset + i_start_1, j)
            i_end_1 = i_end_1 - 1 if j >= i_end_1 else i_end_1
            i_start_1 = i_start_1 + 1 if j < i_start_1 else i_start_1
            offset += 1

    offset = 0
    for i, customer in enumerate(ind2[i_start_2: i_end_2]):
        if customer not in common:
            sample_space2 = list(range(i_start_2)) + list(range(i_end_2, ics - 1))
            j = sample(sample_space2, 1)[0]
            move_from_to(ind2, i - offset + i_start_2, j)
            i_end_2 = i_end_2 - 1 if j >= i_end_2 else i_end_2
            i_start_2 = i_start_2 + 1 if j < i_start_2 else i_start_2
            offset += 1

    # Swap blocks
    aux = ind1[i_start_1:i_end_1]
    ind1[i_start_1:i_end_1] = ind2[i_start_2:i_end_2]
    ind2[i_start_2:i_end_2] = aux


def crossover_charging_operation1(ind1: IndividualType, ind2: IndividualType, index: int, m: int, num_customers: int,
                                  num_cs: int, r: int):
    idt = len(ind1) - m
    ics = idt - 3 * m * r

    offset = int((index - ics) / 3)
    op_index = ics + 3 * offset
    swap_block(ind1, ind2, op_index, op_index + 3)
    return


def crossover_departure_time1(ind1: IndividualType, ind2: IndividualType, index: int, m: int, num_customers: int,
                              num_cs: int, r: int):
    swap_block(ind1, ind2, index, index + 1)
    return


'''
INITIAL POPULATION FUNCTIONS
'''


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
        departure_time_blocks[i] = uniform(-10, 10)
        customers = [i for i in customers if i not in assigned_customers]

    # Charging station blocks
    charging_operation_blocks = []
    for i in range(m * r):
        cust, cs, amount = sample(customers + [-1] * m * r, 1)[0], sample(charging_stations, 1)[0], uniform(20, 30)
        # cust, cs, amount = -1, sample(charging_stations, 1)[0], uniform(20, 30)
        charging_operation_blocks += [cust, cs, amount]

    individual = customer_blocks + charging_operation_blocks + departure_time_blocks
    return individual


def heuristic_population_1(r: int, fleet: Fleet):
    net = fleet.network
    tw = {i: net.nodes[i].time_window_low for i in net.customers}
    sorted_customers = sorted(tw, key=tw.__getitem__)

    max_weight = fleet.vehicles[0].max_payload

    routes = []
    current_route = []
    cum_weight = 0
    for i in sorted_customers:
        d = net.demand(i)
        if i == sorted_customers[-1]:
            current_route.append(i)
            routes.append(current_route)
        elif cum_weight + d <= max_weight:
            current_route.append(i)
            cum_weight += d
        else:
            routes.append(current_route)
            current_route = [i]
            cum_weight = d

    # Fleet size
    m = len(routes)

    # Create blocks
    customer_block = []
    for route in routes:
        customer_block += route + ['|']

    charging_operation_block = []
    for j in range(m * r):
        cust, cs, amount = -1, sample(net.charging_stations, 1)[0], uniform(20, 30)
        charging_operation_block += [cust, cs, amount]

    dep_time_block = [net.nodes[i].time_window_low for i in [j[0] for j in routes]]
    pop = [customer_block + charging_operation_block + dep_time_block]
    return pop, m


def heuristic_population_2(m: int, r: int, fleet: Fleet):
    net = fleet.network
    customers_per_vehicle = int(len(net.customers) / m)

    tw = {i: net.nodes[i].time_window_low for i in net.customers}
    sorted_tw = sorted(tw, key=tw.__getitem__)

    customers_block = []
    for j in range(m):
        if j == m - 1:
            customers_per_vehicle = len(sorted_tw)
        customers_block += [sorted_tw.pop(0) for _ in range(customers_per_vehicle)] + ['|']

    charging_operations_block = []
    sample_space = tuple(net.customers) + tuple([-1] * m * r)
    for j in range(m * r):
        # cust, cs, amount = sample(sample_space, 1)[0], sample(net.charging_stations, 1)[0], uniform(20, 30)
        customer, charging_station, amount = -1, sample(net.charging_stations, 1)[0], uniform(20, 30)
        charging_operations_block += [customer, charging_station, amount]

    departure_times_block = list(np.random.uniform(60 * 10, 60 * 12, m))

    ind = customers_block + charging_operations_block + departure_times_block
    pop = [ind]

    return pop, m


'''
AUXILIARY FUNCTIONS
'''


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


def resize_individual(individual, old_m, new_m, r, customers, charging_stations):
    n = len(customers)
    customer_insertion = ['|'] * (new_m - old_m)
    cs_insertion = []
    for i in range((new_m - old_m) * r):
        cust, cs, amount = sample(customers + (-1,) * (new_m - old_m) * r, 1)[0], sample(charging_stations, 1)[
            0], uniform(20, 30)
        cs_insertion += [cust, cs, amount]
    dep_time_insertion = [individual[-1] for _ in range(new_m - old_m)]
    return individual[:n + old_m] + customer_insertion + cs_insertion + individual[n + old_m:] + dep_time_insertion


'''
MAIN ALGORITHM
'''


def alphaGA(fleet: Fleet, hp: HyperParameters, save_to: str = None, best_ind=None, savefig=False,
            mi: int = None, plot_best_generation=False):
    # OBJECTS
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin, feasible=False, acceptable=False)

    # INIT SIZES
    if mi:
        pop, mh = heuristic_population_2(mi, hp.r, fleet)
        m = mi
    else:
        pop, mh = heuristic_population_1(hp.r, fleet)
        m = mh
        mi = mh

    fleet.resize_fleet(m)
    init_size = len(pop)
    random_inds_num = int(hp.num_individuals / 3)
    mutate_num = hp.num_individuals - init_size - random_inds_num

    # MODIFY HYPER-PARAMETERS
    hp.num_individuals = int(len(fleet.network) * 1.5) + int(len(fleet) * 10) + 50
    hp.max_generations = hp.num_individuals * 3

    # TOOLBOX
    toolbox = base.Toolbox()
    toolbox.register("individual", random_individual, num_customers=len(fleet.network.customers),
                     num_cs=len(fleet.network.charging_stations), m=len(fleet), r=hp.r)
    toolbox.register("evaluate", fitness, fleet=fleet, hp=hp)
    toolbox.register("mate", crossover, fleet=fleet, hp=hp, block_probability=(.33, .33, .33), index=None)
    toolbox.register("mutate", mutate, fleet=fleet, hp=hp, block_probability=(.33, .33, .33), index=None)
    toolbox.register("select", tools.selTournament, tournsize=hp.tournament_size)
    toolbox.register("select_worst", tools.selWorst)
    toolbox.register("decode", decode, fleet=fleet, hp=hp)

    # BEGIN ALGORITHM
    t_init = time.time()
    # pop = [creator.Individual(i) for i in construct_individuals(hp.r, fleet)]

    # Population from first candidates
    pop = [creator.Individual(i) for i in pop]

    # Random population
    if best_ind is None:
        pop += [toolbox.mutate(toolbox.clone(pop[randint(0, init_size - 1)])) for i in range(mutate_num)]
        pop += [creator.Individual(toolbox.individual()) for i in range(random_inds_num)]
    else:
        pop += [toolbox.mutate(toolbox.clone(pop[randint(0, init_size - 1)])) for i in range(mutate_num)]
        pop += [creator.Individual(toolbox.individual()) for i in range(random_inds_num - 1)]
        pop.append(creator.Individual(best_ind))

    # Evaluate the initial population and get fitness of each individual
    for k, ind in enumerate(pop):
        fit, feasible, acceptable = toolbox.evaluate(ind)
        ind.fitness.values = (fit,)
        ind.feasible = feasible
        ind.acceptable = acceptable

    print(f'  Evaluated {len(pop)} individuals')
    bestOfAll = tools.selBest(pop, 1)[0]
    print(f"Best individual  : {bestOfAll}\n Fitness: {bestOfAll.fitness.wvalues[0]} Feasible: {bestOfAll.feasible}")

    # These will save statistics
    cs_capacity = fleet.network.nodes[fleet.network.charging_stations[0]].capacity
    opt_data = GenerationsData([], [], [], [], [], [], fleet, hp, bestOfAll, bestOfAll.feasible, bestOfAll.acceptable,
                               m, cs_capacity)

    print("################  Start of evolution  ################")
    # Begin the evolution
    for g in range(hp.max_generations):
        # A new generation
        print(f"-- Generation {g}/{hp.max_generations} --")
        opt_data.generations.append(g)

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
            if random() < hp.CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Evaluate the individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            fit, feasible, acceptable = toolbox.evaluate(ind)
            ind.fitness.values = (fit,)
            ind.feasible = feasible
            ind.acceptable = acceptable

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
        print(
            f"Best individual  : {bestInd}\n Fitness: {bestInd.fitness.wvalues[0]} Feasible: {bestInd.feasible} Acceptable: {bestInd.acceptable}")

        worstInd = tools.selWorst(pop, 1)[0]
        print(
            f"Worst individual : {worstInd}\n Fitness: {worstInd.fitness.wvalues[0]} Feasible: {worstInd.feasible}  Acceptable: {worstInd.acceptable}")

        print(
            f"Curr. best-of-all: {bestOfAll}\n Fitness: {bestOfAll.fitness.wvalues[0]} Feasible: {bestOfAll.feasible}  Acceptable: {bestOfAll.acceptable}")

        # Statistics
        fits = [sum(ind.fitness.wvalues) for ind in pop]
        mean = np.average(fits)
        std = np.std(fits)

        print(f"Max {max(fits)}")
        print(f"Min {min(fits)}")
        print(f"Avg {mean}")
        print(f"Std {std}")

        opt_data.best_fitness.append(-max(fits))
        opt_data.worst_fitness.append(-min(fits))
        opt_data.average_fitness.append(mean)
        opt_data.std_fitness.append(std)
        opt_data.best_individuals.append(bestInd)

        print()

        if plot_best_generation:
            toolbox.evaluate(bestOfAll)
            fleet.plot_operation_pyplot()
            plt.show()
            input('Press ENTER to evolve...')

    t_end = time.time()
    print("################  End of (successful) evolution  ################")

    algo_time = t_end - t_init
    print('Algorithm time:', algo_time)

    fit, feasible, acceptable = toolbox.evaluate(bestOfAll)
    routes = toolbox.decode(bestOfAll)

    opt_data.bestOfAll = bestOfAll
    opt_data.feasible = feasible
    opt_data.acceptable = acceptable
    opt_data.algo_time = algo_time
    opt_data.fleet = fleet
    opt_data.additional_info = {'mi': mi, 'mh': mh}

    if save_to:
        path = save_to + hp.algorithm_name + f'_fleetsize_{m}/'
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        opt_data.save_opt_data(path, savefig=savefig)
    return routes, opt_data, toolbox
