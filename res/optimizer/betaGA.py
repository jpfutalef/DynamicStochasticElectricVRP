from random import randint, uniform, sample, random
from deap import tools, base, creator
import numpy as np
from typing import Tuple, Dict
import time, os

from res.optimizer.GATools import HyperParameters, GenerationsData, RouteDict, IndividualType, IndicesType
from res.models import Fleet


# FUNCTIONS
def decode(individual: IndividualType, indices: IndicesType, init_state: Dict, fleet: Fleet) -> RouteDict:
    routes = {}
    for id_ev, (i0, i1, i2) in indices.items():
        S = individual[i0:i1]
        L = [0] * len(S)
        x0_offset = individual[i2]

        charging_operations = individual[i1:i2]
        offset = 1
        for i in range(len(S)):
            charging_station, amount = charging_operations[2 * i], charging_operations[2 * i + 1]
            if charging_station != -1:
                # S = S[:i + offset] + [charging_station] + S[i + offset:]
                # L = L[:i + offset] + [amount] + L[i + offset:]
                S.insert(i + offset, charging_station)
                L.insert(i + offset, amount)
                offset += 1

        S = tuple([0] + S + [0])
        L = tuple([0] + L + [0])
        x1_0 = init_state[id_ev].x1_0 + x0_offset
        x2_0 = init_state[id_ev].x2_0
        x3_0 = init_state[id_ev].x3_0

        routes[id_ev] = ((S, L), x1_0, x2_0, x3_0)
    return routes


def mutate(individual: IndividualType, indices: IndicesType, charging_stations: Tuple, hp: HyperParameters) -> None:
    for i in range(hp.mutation_repeat):
        index = random_block_index(indices)
        for id_ev, (i0, i1, i2) in indices.items():
            if i0 <= index <= i2:
                if i0 + 1 >= i1:
                    break
                if i0 <= index < i1:
                    # Case customer
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

                elif i1 <= index < i2:
                    # Case CS
                    i = i1 + 2 * int((index - i1) / 2)
                    sample_space = list(charging_stations) + [-1] * len(charging_stations)
                    individual[i] = sample(sample_space, 1)[0] if randint(0, 1) else individual[i]
                    individual[i + 1] = abs(individual[i + 1] + uniform(-10, 10))

                else:
                    # Case depart time
                    amount = uniform(-20.0, 20.0)
                    individual[i2] = abs(individual[i2] + amount)
                break


def crossover(ind1: IndividualType, ind2: IndividualType, indices: IndicesType, hp: HyperParameters) -> None:
    for i in range(hp.crossover_repeat):
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
    if i0 + 1 >= i1:
        return 0
    elif random() < 0.3:
        return randint(i0, i1 - 1)
    elif random() < 0.8:
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
        i2 = i1 + 2 * ni
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
            # charging_operations.append(sample(sample_space, 1)[0])
            charging_operations.append(-1)
            charging_operations.append(uniform(5, 20))
        depart_time = [uniform(60 * 9, 60 * 12)]
        individual += customer_sequence + charging_operations + depart_time
    return individual


def fitness(individual: IndividualType, indices: IndicesType, init_state: Dict,
            fleet: Fleet, hp: HyperParameters):
    """
    Positive fitness of the individual.
    :param individual: the individual to evaluate
    :param fleet: fleet instance
    :param hp: GA hyper-parameters
    :return: fitness value of the individual (positive)
    """

    # Decode
    routes = decode(individual, indices, init_state, fleet)

    # Set routes
    fleet.set_routes_of_vehicles(routes)

    # Cost
    costs = np.array(fleet.cost_function())

    # Calculate penalization
    feasible, distance, accept = fleet.feasible()

    penalization = distance + hp.hard_penalization if fleet.deterministic else distance

    # Calculate fitness
    fit = np.dot(costs, np.asarray(hp.weights)) + penalization

    return fit, feasible, accept


def individual_from_routes(fleet: Fleet) -> IndividualType:
    routes = {ev_id: (ev.route, ev.x1_0, ev.x2_0, ev.x3_0) for ev_id, ev in fleet.vehicles.items()}
    ind = []
    for (S, L), x10, x20, x30 in routes.values():
        customer_block, CBP1 = [], []
        charging_path = False
        charging_station, soc_increment = None, None

        for Sk, Lk in zip(S[1:-1], L[1:-1]):
            if fleet.network.is_customer(Sk):
                customer_block.append(Sk)
                if charging_path:
                    CBP1[-2:] = [charging_station, soc_increment]
                    charging_path = False
                CBP1 += [-1, uniform(5, 20)]
            elif fleet.network.is_charging_station(Sk):
                charging_station, soc_increment = Sk, Lk
                charging_path = True

        offset_time = 0.
        ind += customer_block + CBP1 + [offset_time]
    return ind


# THE ALGORITHM
def betaGA(fleet: Fleet, hp: HyperParameters, save_to: str = None, best_ind: IndividualType = None,
           savefig=False):
    fleet.assign_customers_in_route()
    customers_to_visit = {ev_id: ev.assigned_customers for ev_id, ev in fleet.vehicles.items()}
    starting_points = {ev_id: (0, 0, fleet.vehicles[ev_id].state_leaving[0, 0],
                                               ev.alpha_up, sum([fleet.network.demand(x)
                                                                 for x in ev.assigned_customers]))
                       for ev_id, ev in fleet.vehicles.items()}
    indices = block_indices(customers_to_visit)

    # Fitness objects
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin, feasible=False, acceptable=False)

    # Toolbox
    toolbox = base.Toolbox()

    toolbox.register("individual", random_individual, customers_per_vehicle=customers_to_visit,
                     charging_stations=fleet.network.charging_stations)
    toolbox.register("evaluate", fitness, indices=indices, init_state=starting_points, fleet=fleet, hp=hp)
    toolbox.register("mate", crossover, indices=indices, hp=hp)
    toolbox.register("mutate", mutate, indices=indices, charging_stations=fleet.network.charging_stations, hp=hp)
    toolbox.register("select", tools.selTournament, tournsize=hp.tournament_size)
    toolbox.register("select_worst", tools.selWorst)
    toolbox.register("decode", decode, indices=indices, init_state=starting_points, fleet=fleet)

    # BEGIN ALGORITHM
    t_init = time.time()

    # Random population
    if best_ind is not None:
        pop = [creator.Individual(best_ind)]
    else:
        pop = []

    pop.append(creator.Individual(individual_from_routes(fleet)))
    pop = pop + [creator.Individual(toolbox.individual()) for i in range(hp.num_individuals - len(pop))]

    # Evaluate the initial population and get fitness of each individual
    for ind in pop:
        fit, feasible, acceptable = toolbox.evaluate(ind)
        ind.fitness.values = (fit,)
        ind.feasible = feasible
        ind.acceptable = acceptable

    print(f'  Evaluated {len(pop)} individuals')
    bestOfAll = tools.selBest(pop, 1)[0]
    print(f"Best individual  : {bestOfAll}\n Fitness: {bestOfAll.fitness.wvalues[0]} Feasible: {bestOfAll.feasible}")

    # These will save statistics
    m = len(fleet)
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
        if hp.elite_individuals:
            best_individuals = list(map(toolbox.clone, tools.selBest(pop, hp.elite_individuals)))

        # Select and clone the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Mutation
        for mutant in offspring:
            if random() < hp.MUTPB:
                toolbox.mutate(mutant)
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
        if hp.elite_individuals:
            pop[:] = best_individuals + pop[:-hp.elite_individuals]

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
    routes = toolbox.decode(bestOfAll)

    opt_data.bestOfAll = bestOfAll
    opt_data.feasible = feasible
    opt_data.acceptable = acceptable
    opt_data.algo_time = algo_time
    opt_data.fleet = fleet

    if save_to:
        path = save_to + hp.algorithm_name + f'_fleetsize_{m}/'
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        opt_data.save_opt_data(path, savefig=savefig)

    return routes, opt_data, toolbox
