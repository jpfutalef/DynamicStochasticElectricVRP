import os
import time
from random import randint, uniform, sample
from typing import Tuple, Dict

import numpy as np
from deap import tools, base, creator

from res.optimizer.GATools import HyperParameters, GenerationsData, Fleet, RouteDict, IndividualType, IndicesType, \
    StartingPointsType, InitialCondition


# FUNCTIONS
def decode(individual: IndividualType, indices: IndicesType, critical_points: StartingPointsType, fleet: Fleet,
           hp: HyperParameters) -> RouteDict:
    routes = {}
    for id_ev, (i0, i1, i2) in indices.items():
        initial_condition = critical_points[id_ev]
        S = individual[i0:i1]
        chg_block = individual[i1:i2]
        offset = individual[i2]

        L = [0] * len(S)
        chg_ops = [(chg_block[3 * i], chg_block[3 * i + 1], chg_block[3 * i + 2]) for i in range(hp.r)]
        for customer, charging_station, amount in chg_ops:
            if customer != -1:
                index = S.index(customer) + 1 if customer != initial_condition.S0 else 0
                if index == len(S):
                    S.insert(index, charging_station)
                    L.insert(index, amount)
                elif S[index] == charging_station:
                    L[index] += amount
                else:
                    S.insert(index, charging_station)
                    L.insert(index, amount)

        S0 = initial_condition.S0
        if S0 == 0:
            # Initial node is the depot
            L0 = initial_condition.L0
            x10 = initial_condition.x1_0 + offset
            x20 = initial_condition.x2_0
        elif critical_points[id_ev].L0 > 0:
            # Initial node is a CS
            L0 = abs(initial_condition.L0 + offset) if abs(initial_condition.L0 + offset) + initial_condition.x2_0 <= fleet.vehicles[id_ev].alpha_up else fleet.vehicles[id_ev].alpha_up
            x10 = initial_condition.x1_0 + fleet.network.spent_time(S0, initial_condition.x2_0, L0)
            x20 = initial_condition.x2_0 + L0
        else:
            # Initial node is a customer
            L0 = initial_condition.L0
            x10 = initial_condition.x1_0 + fleet.network.spent_time(S0, initial_condition.x2_0, L0) + offset
            x20 = initial_condition.x2_0
        x30 = initial_condition.x3_0 - fleet.network.demand(S0)
        S = tuple([S0] + S + [0])
        L = tuple([L0] + L + [0])
        routes[id_ev] = ((S, L), x10, x20, x30)
    return routes


def mutate(individual: IndividualType, indices: IndicesType, starting_points: StartingPointsType,
           customers_to_visit: Dict[int, Tuple[int, ...]], charging_stations: Tuple, allowed_charging_operations=2,
           index=None) -> IndividualType:
    # Choose a random index if not passed
    if index is None:
        index = randint(0, len(individual) - 1)

    # Find block
    for id_ev, (i0, i1, i2) in indices.items():
        if i0 <= index <= i2:
            # One or no customers assigned
            if i0 + 1 >= i1:
                break

            # Case customer
            if i0 <= index < i1:
                case = np.random.random()
                if case <= 0.40:
                    i, j = np.random.randint(i0, i1), np.random.randint(i0, i1)
                    while i == j:
                        j = np.random.randint(i0, i1)
                    swap_elements(individual, i, j)
                elif case < 0.85:
                    i = np.random.randint(i0, i1)
                    individual[i0:i1] = individual[i:i1] + individual[i0:i]
                else:
                    individual[i0:i1] = sample(individual[i0:i1], i1 - i0)

            # Case CS
            elif i1 <= index < i2:
                i = i1 + 3 * int((index - i1) / 3)
                r = allowed_charging_operations
                rr = i1 - i0
                sample_space = individual[i0:i1] + [starting_points[id_ev].S0]*r + [-1]*rr + [individual[i]]*rr
                individual[i] = sample(sample_space, 1)[0]
                individual[i + 1] = sample(charging_stations, 1)[0]
                new_val = abs(individual[i + 2] + np.random.normal(0, 10))
                new_val = new_val if new_val <= 90 else 90
                individual[i + 2] = new_val

            # Case offset of initial condition
            else:
                if starting_points[id_ev].L0:
                    amount = individual[i2] + np.random.normal(-2, 2)
                elif starting_points[id_ev].S0 == 0:
                    amount = individual[i2] + np.random.normal(-2, 2)
                    amount = amount if amount > -15. else -15
                else:
                    amount = abs(individual[i2] + np.random.normal(-1, 1))
                individual[i2] = amount
    return individual


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
        i2 += ni + 3 * allowed_charging_operations

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
    :param indices:
    :param starting_points:
    :param customers_to_visit:
    :param charging_stations:
    :param allowed_charging_operations:
    :return:
    """
    individual = []
    for id_ev, (i0, i1, i2) in indices.items():
        init_point, customers = starting_points[id_ev], customers_to_visit[id_ev]
        customer_block = sample(customers, len(customers))
        sample_space = customers + (init_point.S0,) + (-1,) * allowed_charging_operations
        after_customers = sample(sample_space, allowed_charging_operations)
        charging_block = [0.] * allowed_charging_operations * 3
        for i, customer in enumerate(after_customers):
            charging_block[3 * i] = customer
            charging_block[3 * i + 1] = sample(charging_stations, 1)[0]
            charging_block[3 * i + 2] = uniform(10.0, 90.0)
        if init_point.L0:
            offset_block = [uniform(-8, 8)]
        else:
            offset_block = [uniform(0, 15)]
        individual += customer_block + charging_block + offset_block
    return individual


def fitness(individual: IndividualType, indices: IndicesType, critical_points: StartingPointsType,
            fleet: Fleet, hp: HyperParameters):
    """
    Calculates fitness of individual.
    """

    # Decode
    m = len(fleet.vehicles)
    routes = decode(individual, indices, critical_points, fleet, hp)

    # Set routes
    reaching_states = {id_ev: (r.x1_0, r.x2_0, r.x3_0) for id_ev, r in critical_points.items()}
    vehicles_pos = [cp.S0 for cp in critical_points.values()]
    init_theta = np.array([1 if i in vehicles_pos else 0 for i in range(len(fleet.network))])
    fleet.set_routes_of_vehicles(routes, reaching_states=reaching_states, init_theta=init_theta)

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


def code(fleet: Fleet, routes_with_critical_point: RouteDict, allowed_charging_operations=2) -> Tuple[
    IndividualType, StartingPointsType]:
    ind = []
    critical_points = {}
    for id_ev, route in routes_with_critical_point.items():
        # Get the route starting from the critical point
        (S, L), x1, x2, x3 = route

        # Save critical points
        critical_points[id_ev] = InitialCondition(S[0], L[0], x1, x2, x3)

        # Create individual from ahead route
        customer_sequence = [x for x in S[1:] if fleet.network.isCustomer(x)]
        charging_operations, charging_operations_count = [], 0
        for k, LK in enumerate(L[1:]):
            if LK > 0.:
                charging_operations += [S[k], S[k + 1], LK]
                charging_operations_count += 1

        for i in range(allowed_charging_operations - charging_operations_count):
            charging_operations += [-1, sample(fleet.network.charging_stations, 1)[0], uniform(10, 90)]

        ind += customer_sequence + charging_operations + [0.]
        fleet.vehicles[id_ev].assigned_customers = tuple(customer_sequence)

    return ind, critical_points


# THE ALGORITHM
def onGA(fleet: Fleet, hp: HyperParameters, critical_points: StartingPointsType, save_to: str = None,
         best_ind: IndividualType = None, savefig=False):
    customers_to_visit = {ev_id: fleet.vehicles[ev_id].assigned_customers for ev_id in fleet.vehicles_to_route}
    indices = block_indices(customers_to_visit, hp.r)

    # Fitness objects
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin, feasible=False, acceptable=False)

    # Toolbox
    toolbox = base.Toolbox()

    toolbox.register("individual", random_individual, indices=indices, starting_points=critical_points,
                     customers_to_visit=customers_to_visit, charging_stations=fleet.network.charging_stations,
                     allowed_charging_operations=hp.r)
    toolbox.register("evaluate", fitness, indices=indices, critical_points=critical_points, fleet=fleet, hp=hp)
    toolbox.register("mate", crossover, indices=indices, allowed_charging_operations=hp.r, index=None)
    toolbox.register("mutate", mutate, indices=indices, starting_points=critical_points,
                     customers_to_visit=customers_to_visit, charging_stations=fleet.network.charging_stations,
                     allowed_charging_operations=hp.r, index=None)
    toolbox.register("select", tools.selTournament, tournsize=hp.tournament_size)
    toolbox.register("select_worst", tools.selWorst)
    toolbox.register("decode", decode, indices=indices, critical_points=critical_points, fleet=fleet, hp=hp)

    # BEGIN ALGORITHM
    t_init = time.time()

    # Random population
    if best_ind is None:
        pop = [creator.Individual(toolbox.individual()) for i in range(hp.num_individuals)]
    else:
        pop = [creator.Individual(best_ind)]
        pop += [creator.Individual(toolbox.individual()) for i in range(hp.num_individuals - 1)]

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
                               len(fleet), cs_capacity)
    print("################  Start of evolution  ################")
    # Begin the evolution
    for g in range(hp.max_generations):
        # A new generation
        print(f"-- Generation {g}/{hp.max_generations} --")
        opt_data.generations.append(g)

        # Select the best individuals, if given
        if hp.keep_best:
            best_individuals = list(map(toolbox.clone, tools.selBest(pop, hp.keep_best)))

        # Select and clone the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Mutation
        for mutant in offspring:
            if np.random.random() < hp.MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < hp.MUTPB:
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

        opt_data.best_fitness.append(-max(fits))
        opt_data.worst_fitness.append(-min(fits))
        opt_data.average_fitness.append(mean)
        opt_data.std_fitness.append(std)
        opt_data.best_individuals.append(bestInd)

        print()

    t_end = time.time()
    print("################  End of (successful) evolution  ################")

    algo_time = t_end - t_init
    print('Algorithm time:', algo_time)

    fit, feasible, acceptable = toolbox.evaluate(bestOfAll)
    fit, feasible, acceptable = toolbox.evaluate(bestOfAll)
    routes = toolbox.decode(bestOfAll)

    opt_data.bestOfAll = bestOfAll
    opt_data.feasible = feasible
    opt_data.acceptable = acceptable
    opt_data.algo_time = algo_time
    opt_data.fleet = fleet

    if save_to:
        path = save_to
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        opt_data.save_opt_data(path, savefig=savefig)

    for r in routes.values():
        L = r[0][1]
        if sum([1 for Lk in L if Lk < 0]):
            print('negative')
    return routes, opt_data, toolbox
