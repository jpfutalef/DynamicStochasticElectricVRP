import os, time
from random import randint, uniform, sample
from typing import Tuple, Dict, NamedTuple
import matplotlib.pyplot as plt
import numpy as np
from deap import tools, base, creator

import res.optimizer.GATools as GA
import res.models.NonLinearOps as NL
import res.models.Fleet as Fleet
import res.models.Network as Network


class CriticalState(NamedTuple):
    j_crit: int
    S0: int
    L0: float
    x1_0: float
    x2_0: float
    x3_0: float


CriticalStateTuple = Tuple[CriticalState, Tuple[int, ...], Tuple[float, ...]]
CriticalStateDict = Dict[int, CriticalStateTuple]


# FUNCTIONS
def decode(individual: GA.IndividualType, indices: GA.IndicesType, hp: GA.OnGA_HyperParameters,
           critical_states: CriticalStateDict) -> GA.RouteDict:
    routes = {}
    for id_ev, (i0, i1, i2) in indices.items():
        critical_state, _, _ = critical_states[id_ev]

        S = individual[i0:i1]
        L = [0.] * len(S)

        chg_block = individual[i1:i2]
        chg_ops = [(chg_block[3 * i], chg_block[3 * i + 1], chg_block[3 * i + 2]) for i in range(hp.r) if
                   chg_block[3 * i] != -1]
        for customer, charging_station, amount in chg_ops:
            index = 0 if customer == critical_state.S0 else S.index(customer) + 1
            S.insert(index, charging_station)
            if index < len(S) - 1 and S[index] == S[index+1]:
                S = S[:index]+ S[index+1:]
                L[index] += amount
            else:
                L.insert(index, amount)

        S0 = critical_state.S0
        L0 = critical_state.L0
        x1_0 = critical_state.x1_0
        x2_0 = critical_state.x2_0
        x3_0 = critical_state.x3_0
        offset = individual[i2] if abs(individual[i2]) > 1e-10 else 0.
        post_wt0 = 0.

        if not S0:
            x1_0 = NL.saturate(x1_0 + offset, x1_0 - hp.offset_time_depot)

        elif critical_state.L0:
            L0 = critical_state.L0 + offset

        else:
            post_wt0 = offset

        S = tuple([S0] + S + [0])
        L = tuple([L0] + L + [0])

        if S[1] == S0:
            S = S[1:]
            L = (L[0] + L[1],) + L[2:]

        routes[id_ev] = (S, L, x1_0, x2_0, x3_0, post_wt0)
    return routes


def mutate(individual: GA.IndividualType, indices: GA.IndicesType, fleet: Fleet.Fleet, hp: GA.OnGA_HyperParameters,
           critical_states: CriticalStateDict, index=None) -> GA.IndividualType:
    # Choose a random index if not passed
    if index is None:
        index = randint(0, len(individual) - 1)

    # Find block
    for id_ev, (i0, i1, i2) in indices.items():
        critical_state, _, _ = critical_states[id_ev]
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
                    GA.swap_elements(individual, i, j)
                elif case < 0.85:
                    i = np.random.randint(i0, i1)
                    individual[i0:i1] = individual[i:i1] + individual[i0:i]
                else:
                    individual[i0:i1] = sample(individual[i0:i1], i1 - i0)

            # Case CS
            elif i1 <= index < i2:
                i = i1 + 3 * int((index - i1) / 3)
                if fleet.network.is_charging_station(critical_state.S0) or not critical_state.S0:
                    sample_space = individual[i0:i1] + [-1] + [individual[i]]
                else:
                    sample_space = individual[i0:i1] + [-1] + [individual[i]] + [critical_state.S0]
                individual[i] = sample(sample_space, 1)[0]
                individual[i + 1] = sample(fleet.network.charging_stations, 1)[0]
                new_val = abs(individual[i + 2] + np.random.normal(0, 10))
                new_val = new_val if new_val <= 90 else 90
                individual[i + 2] = new_val

            # Case offset of initial condition
            else:
                if critical_state.L0:
                    amount = individual[i2] + np.random.normal(0, 2)
                elif critical_state.S0 == 0:
                    amount = individual[i2] + np.random.normal(0, 8 * 60)
                    amount = amount if amount > -10 * 60. else -10 * 60
                else:
                    amount = abs(individual[i2] + np.random.normal(0, 60))
                individual[i2] = amount
    return individual


def crossover(ind1: GA.IndividualType, ind2: GA.IndividualType, indices: GA.IndicesType, hp: GA.OnGA_HyperParameters,
              index=None) -> None:
    # Choose a random index if not passed
    if index is None:
        index = randint(0, len(ind1))

    # Find block
    for id_ev, (i0, i1, i2) in indices.items():
        if i0 <= index <= i2:
            # Case customer
            if i0 <= index < i1:
                GA.swap_block(ind1, ind2, i0, i1)
                return

            # Case CS
            elif i1 <= index < i2:
                GA.swap_block(ind1, ind2, i1, i2)
                return

            # Case offset
            else:
                GA.swap_block(ind1, ind2, i2, i2)


def block_indices(fleet: Fleet.Fleet, hp: GA.OnGA_HyperParameters) -> GA.IndicesType:
    """
    Creates the indices of sub blocks.
    :param fleet:
    :param hp:
    :return: tuples list with indices [(i0, j0), (i1, j1),...] where iN represent location of the beginning of customers_per_vehicle
    block and jN location of the beginning of charging operations block of evN in the individual, respectively.
    """
    indices = {}

    i0 = 0
    i1 = 0
    i2 = 0
    for id_vehicle, ev in fleet.vehicles.items():
        ni = len(ev.assigned_customers)
        i1 += ni
        i2 += ni + 3 * hp.r

        indices[id_vehicle] = (i0, i1, i2)

        i0 += ni + 3 * hp.r + 1
        i1 += 3 * hp.r + 1
        i2 += 1

    return indices


def random_individual(indices: GA.IndicesType, fleet: Fleet.Fleet, hp: GA.OnGA_HyperParameters,
                      critical_states: CriticalStateDict,):
    individual = []
    for id_ev, (i0, i1, i2) in indices.items():
        critical_state, _, _ = critical_states[id_ev]
        customers = fleet.vehicles[id_ev].assigned_customers
        customer_block = sample(customers, len(customers))
        sample_space = customer_block + [critical_state.S0] + [-1] * hp.r
        after_customers = sample(sample_space, hp.r)
        charging_block = [0.] * hp.r * 3
        for i, customer in enumerate(after_customers):
            charging_block[3 * i] = customer
            charging_block[3 * i + 1] = sample(fleet.network.charging_stations, 1)[0]
            charging_block[3 * i + 2] = uniform(10.0, 90.0)
        if critical_state.L0:
            offset_block = [uniform(-8, 8)]
        elif not critical_state.S0:
            offset_block = [uniform(-hp.offset_time_depot, 0.2 * hp.offset_time_depot)]
        else:
            offset_block = [uniform(0, 15 * 60)]
        individual += customer_block + charging_block + offset_block
    return individual


def fitness(individual: GA.IndividualType, indices: GA.IndicesType, fleet: Fleet.Fleet,
            hp: GA.OnGA_HyperParameters, critical_states: CriticalStateDict):
    # Decode
    routes = decode(individual, indices, hp, critical_states)

    # Obtain theta vector
    init_theta = np.zeros(len(fleet.network))
    for pos in [crit_state.S0 for crit_state, _, _ in critical_states.values()]:
        init_theta[pos] += 1

    # Set routes
    fleet.set_routes_of_vehicles(routes)

    # Iterate
    fleet.iterate(init_theta=init_theta)

    # Cost
    costs = np.array(fleet.cost_function())

    # Check feasibility and penalization
    feasible, distance, accept = fleet.feasible()
    penalization = 0
    if not feasible:
        penalization = distance + hp.hard_penalization

    # Calculate fitness
    fit = np.dot(costs, np.asarray(hp.weights)) + penalization

    return fit, feasible, accept


def code(fleet: Fleet.Fleet, critical_states: CriticalStateDict, hp: GA.OnGA_HyperParameters) -> GA.IndividualType:
    # Container of candidate individual
    ind = []

    # Iterate old solution
    for id_ev, (critical_state, S, L) in critical_states.items():
        # Get block of customers
        customer_sequence = [x for x in S[1:] if fleet.network.is_customer(x)]

        # Get the ROB
        charging_operations = []
        charging_operations_count = 0
        last_customer = S[0]
        for k, (Sk, Lk) in enumerate(zip(S[1:], L[1:])):
            if fleet.network.is_customer(Sk) or fleet.network.is_depot(Sk):
                last_customer = Sk
            else:
                charging_operations += [last_customer, Sk, Lk]
                charging_operations_count += 1

        # If there is more space available in the ROB, fill in with random recharging operations
        for i in range(hp.r - charging_operations_count):
            charging_operations += [-1, sample(fleet.network.charging_stations, 1)[0], uniform(5, 90)]

        # Append blocks to individual with zero offset
        ind += customer_sequence + charging_operations + [0.]

    return ind


# THE ALGORITHM
def onGA(fleet: Fleet.Fleet, hp: GA.OnGA_HyperParameters, critical_states: CriticalStateDict, save_to: str = None,
         best_ind: GA.IndividualType = None, savefig=False):
    # hp.num_individuals = 4 * len(fleet) + 2 * sum([len(fleet.vehicles[ev_id].assigned_customers) for
    #                                                ev_id in fleet.vehicles_to_route]) + 15
    # hp.max_generations = 3 * hp.num_individuals + 30

    customers_to_visit = {ev_id: fleet.vehicles[ev_id].assigned_customers for ev_id in fleet.vehicles_to_route}
    indices = block_indices(fleet, hp)

    # Fitness objects
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin, feasible=False, acceptable=False)

    # Toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", random_individual, indices=indices, fleet=fleet, hp=hp,
                     critical_states=critical_states)
    toolbox.register("evaluate", fitness, indices=indices, fleet=fleet, hp=hp, critical_states=critical_states)
    toolbox.register("decode", decode, indices=indices, hp=hp, critical_states=critical_states)
    toolbox.register("mate", crossover, indices=indices, hp=hp, index=None)
    toolbox.register("mutate", mutate, indices=indices, fleet=fleet, hp=hp, critical_states=critical_states, index=None)
    toolbox.register("select", tools.selTournament, tournsize=hp.tournament_size)
    toolbox.register("select_worst", tools.selWorst)

    # OPTIMIZATION ITERATIONS CONTAINER
    generations_data_container = None
    if save_to:
        generations_data_container = GA.GenerationsData()

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

    print("################  Start of evolution  ################")
    # Begin the evolution
    for g in range(hp.max_generations):
        # Select the best individuals, if given
        if hp.elite_individuals:
            best_individuals = list(map(toolbox.clone, tools.selBest(pop, hp.elite_individuals)))

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

        # GUI info
        worstInd = tools.selWorst(pop, 1)[0]
        fits = [sum(ind.fitness.wvalues) for ind in pop]
        mean = np.average(fits)
        std = np.std(fits)
        to_print = f"""-- Generation {g}/{hp.max_generations} --
Evaluated {len(invalid_ind)} individuals during genetic operations
Curr. best-of-all: {bestOfAll}\n Fitness: {bestOfAll.fitness.wvalues[0]} Feasible: {bestOfAll.feasible}
Worst individual : {worstInd}\n Fitness: {worstInd.fitness.wvalues[0]} Feasible: {worstInd.feasible}

Pop. Max: {max(fits)}
Pop. Min: {min(fits)}
Pop. Avg: {mean}
Pop. Std: {std}"""
        print(to_print, end="\n\r")

        if save_to:
            generations_data_container.generation.append(g)
            generations_data_container.best_fitness.append(-max(fits))
            generations_data_container.worst_fitness.append(-min(fits))
            generations_data_container.feasible.append(bestOfAll.feasible)
            generations_data_container.average_fitness.append(mean)
            generations_data_container.std_fitness.append(std)

        deb = False
        if deb:
            toolbox.evaluate(bestOfAll)
            fleet.plot_operation_pyplot()
            plt.show()

    t_end = time.time()
    print("################  End of (successful) evolution  ################")

    algo_time = t_end - t_init
    print('Algorithm time:', algo_time)

    fit, feasible, acceptable = toolbox.evaluate(bestOfAll)
    routes = toolbox.decode(bestOfAll)
    toolbox.evaluate(bestOfAll)
    report = GA.OptimizationData(fleet, hp, feasible, len(fleet), algo_time)
    if save_to:
        os.makedirs(save_to)
        generations_data_container.save(save_to)
        report.save(save_to)
    return routes, toolbox, report
