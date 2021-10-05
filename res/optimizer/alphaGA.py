import os
import time
from random import randint, uniform, sample, random
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from deap import tools, base, creator

from res.optimizer.GATools import AlphaGA_HyperParameters, GenerationsData, OptimizationData, RouteDict, IndividualType
from res.models import Fleet

'''
MAIN FUNCTIONS
'''


def decode(individual: IndividualType, fleet: Fleet, hp: AlphaGA_HyperParameters) -> RouteDict:
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
    for i, (S, L, departure_time) in enumerate(zip(customer_sequences, charging_sequences, depart_times)):
        S = tuple([0] + S + [0])
        L = tuple([0.] + L + [0.])
        departure_soc = hp.alpha_up
        departure_payload = sum([fleet.network.demand(x) for x in S])
        routes[i] = (S, L, departure_time, departure_soc, departure_payload, 0)
    return routes


def fitness(individual: IndividualType, fleet: Fleet, hp: AlphaGA_HyperParameters):
    """
    Positive fitness of the individual.
    :param individual: the individual to evaluate
    :param fleet: fleet instance
    :param hp: GA hyper-parameters
    :return: fitness value of the individual (positive)
    """

    # Decode
    routes = decode(individual, fleet, hp)

    # Set routes
    fleet.set_routes_of_vehicles(routes)

    # Iterate
    fleet.iterate()

    # Cost
    costs = np.array(fleet.cost_function())  # cost_tt, cost_ec, cost_chg_time, cost_chg_cost

    # Calculate penalization
    feasible, distance, accept = fleet.feasible()

    if distance > 0.:
        penalization = distance + hp.hard_penalization
    else:
        penalization = 0

    # Calculate fitness
    fit = np.dot(costs, np.asarray(hp.weights)) + penalization

    return fit, feasible, accept


def mutate(individual: IndividualType, fleet: Fleet, hp: AlphaGA_HyperParameters, block_probability=(.33, .33, .33),
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


def crossover(ind1: IndividualType, ind2: IndividualType, fleet: Fleet, hp: AlphaGA_HyperParameters,
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

    a0 = list(range(1, num_customers + 1)) + [-1] * num_customers
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
    individual[index] += np.random.normal(60, 3600)


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


def heuristic_population_1(r: int, fleet: Fleet, fill_up_to=1.0):
    """
    Calculate the minimum number of EVs and first route candidates
    @param r: allowed charging operations per vehicle
    @param fleet:
    @return:
    """
    net = fleet.network
    tw = {i: net.nodes[i].time_window_low for i in net.customers}
    sorted_customers = sorted(tw, key=tw.__getitem__)

    max_weight = fleet.vehicles[0].max_payload * fill_up_to

    routes = []
    current_route = []
    cum_weight = 0
    for i in sorted_customers:
        d = net.demand(i)
        if cum_weight + d >= max_weight:
            routes.append(current_route)
            current_route = [i]
            cum_weight = 0
        else:
            current_route.append(i)
            cum_weight += d

    if current_route:
        routes.append(current_route)

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

    departure_times_block = list(np.random.uniform(60 * 60 * 10, 60 * 60 * 12, m))

    ind = customers_block + charging_operations_block + departure_times_block
    pop = [ind]

    return pop, m


def heuristic_population_3(r: int, fleet: Fleet.GaussianFleet, fill_up_to=1.0, additional_vehicles=1):
    net = fleet.network
    tw = {i: net.nodes[i].time_window_low for i in net.customers}
    sorted_customers = sorted(tw, key=tw.__getitem__)

    max_weight = fleet.vehicles[0].max_payload * fill_up_to
    max_tour_time = fleet.vehicles[0].max_tour_duration

    routes = []
    current_route = []
    cum_weight, cum_av_time, t = 0, 0, 0
    for k, i in enumerate(sorted_customers):
        d = net.demand(i)
        if k > 0:
            j = sorted_customers[k - 1]
            edge = net.edges[i][j]
            t = edge.length / np.min(edge.velocity)
        if cum_weight + d >= max_weight or cum_av_time + t >= max_tour_time :
            routes.append(current_route)
            current_route = [i]
            cum_weight = 0
            cum_av_time = 0
        else:
            current_route.append(i)
            cum_weight += d
            cum_av_time += t

    # Check if there are residual customers after loop
    if current_route:
        routes.append(current_route)

    # Fill up to fleet size if necessary
    [routes.append([]) for _ in range(additional_vehicles)]
    fleet.resize_fleet(len(routes))

    # Create blocks
    customer_block = []
    for route in routes:
        customer_block += route + ['|']

    charging_operation_block = []
    for j in range(r * len(fleet)):
        customer, charging_station, amount = -1, sample(net.charging_stations, 1)[0], uniform(20, 30)
        charging_operation_block += [customer, charging_station, amount]

    departure_time_block = [net.nodes[r[0]].time_window_low if r else np.random.uniform(7 * 60 * 60, 13 * 60 * 60) for r in
                            routes]
    initial_population = [customer_block + charging_operation_block + departure_time_block]
    return initial_population


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


def alphaGA(fleet: Fleet, hp: AlphaGA_HyperParameters, save_to: str = None, init_pop=None, instance_path=None):
    # OBJECTS
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin, feasible=False, acceptable=False)

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

    # OPTIMIZATION ITERATIONS CONTAINER
    generations_data_container = None
    if save_to:
        generations_data_container = GenerationsData()

    # BEGIN ALGORITHM
    t_init = time.time()

    # Population from first candidates
    random_inds_num = int(hp.num_individuals / 3)
    init_size = len(init_pop)
    mutate_num = hp.num_individuals - init_size - random_inds_num
    pop = [creator.Individual(i) for i in init_pop]

    # Random population
    pop += [toolbox.mutate(toolbox.clone(pop[randint(0, init_size - 1)])) for i in range(mutate_num)]
    pop += [creator.Individual(toolbox.individual()) for _ in range(random_inds_num)]

    # Evaluate the initial population and get fitness of each individual
    for k, ind in enumerate(pop):
        fit, feasible, acceptable = toolbox.evaluate(ind)
        ind.fitness.values = (fit,)
        ind.feasible = feasible
        ind.acceptable = acceptable

    print(f'  Evaluated {len(pop)} individuals')
    bestOfAll = tools.selBest(pop, 1)[0]
    print(f"Best individual  : {bestOfAll}\n Fitness: {bestOfAll.fitness.wvalues[0]} Feasible: {bestOfAll.feasible}")

    # Store best fitness average in previous generations
    best_fitness_criterion = 10 * len(fleet.network)
    last_best_fitnesses = [0] * best_fitness_criterion

    # Block probabilities
    block_probabilities = (1/3, 1/3, 1/3)

    # Begin the evolution
    print("################  Start of evolution  ################")
    for g in range(hp.max_generations):
        # Select the best individuals, if given
        if hp.elite_individuals:
            best_individuals = list(map(toolbox.clone, tools.selBest(pop, hp.elite_individuals)))

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

        # The population is entirely replaced by a sorted offspring
        pop[:] = offspring
        pop[:] = tools.selBest(pop, len(pop))

        # Insert best individuals from previous generation
        if hp.elite_individuals:
            pop[:] = best_individuals + pop[:-hp.elite_individuals]

        # Update best individual
        bestInd = tools.selBest(pop, 1)[0]
        if bestInd.fitness.wvalues[0] > bestOfAll.fitness.wvalues[0]:
            bestOfAll = bestInd

        # Real-time info
        worstInd = tools.selWorst(pop, 1)[0]

        # Statistics
        fits = [sum(ind.fitness.wvalues) for ind in pop]
        mean = np.average(fits)
        std = np.std(fits)

        to_print = f"""
------- Generation {g+1}/{hp.max_generations} -------
Evaluated {len(invalid_ind)} individuals during genetic operations
Curr. best-of-all: {bestOfAll}
    Fitness: {bestOfAll.fitness.wvalues[0]}
    Feasible: {bestOfAll.feasible}
Worst individual : {worstInd}
    Fitness: {worstInd.fitness.wvalues[0]}
    Feasible: {worstInd.feasible}

Population Max: {max(fits)}
Population Min: {min(fits)}
Population Avg: {mean}
Population Std: {std}
"""

        print(to_print, end="\r")

        deb = False
        if deb:
            toolbox.evaluate(bestOfAll)
            fleet.plot_operation_pyplot()
            plt.show()

        if save_to:
            generations_data_container.generation.append(g)
            generations_data_container.best_fitness.append(-max(fits))
            generations_data_container.worst_fitness.append(-min(fits))
            generations_data_container.feasible.append(bestOfAll.feasible)
            generations_data_container.average_fitness.append(mean)
            generations_data_container.std_fitness.append(std)

        if g >= best_fitness_criterion:
            last_best_fitnesses.append(bestOfAll.fitness.wvalues[0])
            last_best_fitnesses.pop(0)

            if np.std(last_best_fitnesses) < 1e-10:
                break

        else:
            last_best_fitnesses[g] = bestOfAll.fitness.wvalues[0]

    t_end = time.time()
    print("################  End of (successful) evolution  ################")

    algo_time = t_end - t_init
    print('Algorithm time:', algo_time)

    fit, feasible, acceptable = toolbox.evaluate(bestOfAll)
    routes = toolbox.decode(bestOfAll)

    if save_to:
        os.makedirs(save_to)
        report = OptimizationData(fleet, hp, feasible, len(fleet), algo_time, fit, bestOfAll,
                                  instance_path=instance_path)
        generations_data_container.save(save_to)
        report.save(save_to)

    return routes, toolbox
