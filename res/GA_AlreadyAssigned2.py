from random import randint, uniform, sample, random
from res.GATools import *


# FUNCTIONS
def decode(individual: IndividualType, indices: IndicesType, init_state: StartingPointsType, fleet: Fleet) -> RouteDict:
    routes = {}
    for id_ev, (i0, i1, i2) in indices.items():
        S = individual[i0:i1]
        charging_operations = individual[i1:i2]
        x0_offset = individual[i2]

        L = [0] * len(S)
        offset = 1
        for i, _ in enumerate(S):
            charging_station, amount = charging_operations[2 * i], charging_operations[2 * i + 1]
            if charging_station != -1:
                S = S[:i + offset] + [charging_station] + S[i + offset:]
                L = L[:i + offset] + [amount] + L[i + offset:]
                offset += 1
        S = tuple([0] + S + [0])
        L = tuple([0] + L + [0])
        x1_0 = init_state[id_ev].x1_0 + x0_offset
        routes[id_ev] = ((S, L), x1_0, init_state[id_ev].x2_0, init_state[id_ev].x3_0)
    return routes


def mutate(individual: IndividualType, indices: IndicesType, charging_stations: Tuple, repeat=1) -> None:
    for i in range(repeat):
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
                    sample_space = [-1] * len(charging_stations) + list(charging_stations)
                    individual[i] = sample(sample_space, 1)[0]
                    individual[i + 1] = abs(individual[i + 1] + uniform(-10, 10))

                else:
                    # Case depart time
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
    routes = decode(individual, indices, init_state, fleet)

    # Set routes
    fleet.set_routes_of_vehicles(routes)

    # Cost
    costs = np.array(fleet.cost_function())

    # Check if the solution is feasible
    feasible, penalization, accept = fleet.feasible()

    # penalization
    if not feasible:
        penalization += penalization_constant

    if not accept:
        penalization = penalization ** 2

    fit = np.dot(costs, np.asarray(weights)) + penalization

    return fit, feasible


def individual_from_routes(routes: RouteDict, fleet: Fleet) -> IndividualType:
    ind = []
    for (Sk, Lk), x10, x20, x3 in routes.values():
        cust, chg_ops = [], []
        for node, chg_amount in zip(Sk[1:-1], Lk[1:-1]):
            if chg_amount > 0:
                if chg_ops:
                    chg_ops[-2:] = [node, chg_amount]
                else:
                    chg_ops = [node, chg_amount]
            else:
                cust.append(node)
                chg_ops += [-1, uniform(5, 20)]
        # dep_time = x10 - fleet.network.nodes[Sk[1]].time_window_low if fleet.network.isCustomer(Sk[1]) else 6 * 60
        dep_time = 0.
        ind += cust + chg_ops + [dep_time]
    return ind


# THE ALGORITHM
def optimal_route_assignation(fleet: Fleet, hp: HyperParameters, save_to: str = None, best_ind: IndividualType = None,
                              savefig=False, bf=0.0):
    customers_to_visit = {ev_id: ev.assigned_customers for ev_id, ev in fleet.vehicles.items()}
    starting_points = {ev_id: InitialCondition(0, 0, fleet.vehicles[ev_id].state_leaving[0, 0],
                                               ev.alpha_up, sum([fleet.network.demand(x)
                                                                 for x in ev.assigned_customers]))
                       for ev_id, ev in fleet.vehicles.items()}
    indices = block_indices(customers_to_visit)

    # Fitness objects
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin, feasible=False)

    # Toolbox
    toolbox = base.Toolbox()

    toolbox.register("individual", random_individual, customers_per_vehicle=customers_to_visit,
                     charging_stations=fleet.network.charging_stations)
    toolbox.register("evaluate", fitness, fleet=fleet, indices=indices, init_state=starting_points, weights=hp.weights,
                     penalization_constant=hp.penalization_constant)
    toolbox.register("mate", crossover, indices=indices, repeat=hp.crossover_repeat)
    toolbox.register("mutate", mutate, indices=indices, charging_stations=fleet.network.charging_stations,
                     repeat=hp.mutation_repeat)
    toolbox.register("select", tools.selTournament, tournsize=hp.tournament_size)
    toolbox.register("select_worst", tools.selWorst)
    toolbox.register("decode", decode, indices=indices, init_state=starting_points, fleet=fleet)

    # BEGIN ALGORITHM
    t_init = time.time()

    # Random population
    if best_ind is None:
        pop = [creator.Individual(toolbox.individual()) for i in range(hp.num_individuals)]
    else:
        pop = [creator.Individual(toolbox.individual()) for i in range(hp.num_individuals - 1)]
        pop.append(creator.Individual(best_ind))

    # Evaluate the initial population and get fitness of each individual
    for ind in pop:
        fit, feasible = toolbox.evaluate(ind)
        ind.fitness.values = (fit,)
        ind.feasible = feasible

    print(f'  Evaluated {len(pop)} individuals')
    bestOfAll = tools.selBest(pop, 1)[0]
    print(f"Best individual  : {bestOfAll}\n Fitness: {bestOfAll.fitness.wvalues[0]} Feasible: {bestOfAll.feasible}")

    # These will save statistics
    opt_data = OptimizationIterationsData([], [], [], [], [], [], fleet, hp, bestOfAll, bestOfAll.feasible)
    print(-bf, bestOfAll.fitness.wvalues[0])
    toolbox.evaluate(best_ind)
    routes = toolbox.decode(best_ind)

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

    fit, feasible = toolbox.evaluate(bestOfAll)
    routes = toolbox.decode(bestOfAll)

    opt_data.bestOfAll = bestOfAll
    opt_data.feasible = feasible
    opt_data.algo_time = algo_time

    if save_to:
        try:
            os.mkdir(save_to)
        except FileExistsError:
            pass
        opt_data.save_opt_data(save_to, method='ASSIGNED', savefig=savefig)
    return routes, fleet, bestOfAll, feasible, toolbox, opt_data
