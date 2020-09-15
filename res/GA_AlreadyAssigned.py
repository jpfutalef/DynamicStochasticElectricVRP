from random import randint, uniform, sample, random

from res.GATools import *


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
           customers_to_visit: Dict[int, Tuple[int, ...]], charging_stations: Tuple,
           allowed_charging_operations=2, index=None) -> None:
    # Choose a random index if not passed
    if index is None:
        # index = randint(0, len(individual))
        index = random_block_index(indices)

    # repeat = allowed_charging_operations
    # repeat = len(indices)
    repeat = 1
    for i in range(repeat):
        # Do the following
        for id_ev, (i0, i1, i2) in indices.items():
            # find block
            if i0 <= index <= i2:
                # Case customer
                if i0 <= index < i1:
                    case = random()
                    if case <= 1.0:
                        i = randint(i0, i1 - 1)
                        j = randint(i0, i1 - 1)
                        while i == j:
                            j = randint(i0, i1 - 1)
                        swap_elements(individual, i, j)
                    elif case < 0.8:
                        i = randint(i0, i1 - 1)
                        c1 = individual[i0:i]
                        c2 = individual[i:i1]
                        individual[i0:i1] = c2 + c1
                    else:
                        new_cust = sample(individual[i0:i1], i1 - i0)
                        individual[i0:i1] = new_cust

                # Case CS
                elif i1 <= index < i2:
                    # Find operation sub-block
                    for j in range(allowed_charging_operations):
                        if i1 + j * 3 <= index <= i1 + j * 3 + 2:
                            # Choose if making a charging operation
                            if randint(0, 1):
                                # Choose a customer node randomly
                                sample_space = (starting_points[id_ev].S0,) + customers_to_visit[id_ev]
                                count = 0
                                while count < allowed_charging_operations:
                                    customer = sample(sample_space, 1)[0]
                                    # Ensure customer has not been already chosen
                                    if customer not in [individual[i1 + 3 * x] for x in
                                                        range(allowed_charging_operations)]:
                                        break
                                    count += 1
                                if count == allowed_charging_operations:
                                    individual[i1 + 3 * j] = -1
                                else:
                                    individual[i1 + 3 * j] = customer

                            else:
                                individual[i1 + 3 * j] = -1

                            # Choose a random CS anyways
                            individual[i1 + 3 * j + 1] = sample(charging_stations, 1)[0]

                            # Change amount anyways
                            # amount = uniform(5, 90)
                            # individual[i1 + 3 * j + 2] = amount
                            amount = uniform(-10, 10)
                            new_val = abs(individual[i1 + 3 * j + 2] + float(f"{amount:.2f}"))
                            new_val = new_val if new_val <= 90 else 90
                            individual[i1 + 3 * j + 2] = new_val
                            break

                # Case depart time
                else:
                    amount = uniform(-20.0, 20.0)
                    individual[i2] = abs(individual[i2] + amount)
                    '''
                    amount = uniform(60.0*6, 60.0*20)
                    individual[i2] = amount
                    '''
                break
        # index = randint(0, len(individual))
        index = random_block_index(indices)


def crossover(ind1: IndividualType, ind2: IndividualType, indices: IndicesType, allowed_charging_operations=2,
              index=None) -> None:
    # Choose a random index if not passed
    if index is None:
        # index = randint(0, len(ind1))
        index = random_block_index(indices)

    # repeat = allowed_charging_operations
    # repeat = len(indices)
    repeat = 1
    for i in range(repeat):
        for id_ev, (i0, i1, i2) in indices.items():
            if i0 <= index <= i2:
                # Case customer
                if i0 <= index < i1:
                    swap_block(ind1, ind2, i0, i1)

                # Case CS
                elif i1 <= index < i2:
                    swap_block(ind1, ind2, i1, i2)

                # Case depart time
                else:
                    swap_block(ind1, ind2, i2, i2)

                break

        index = random_block_index(indices)


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
        return sample(range(i1, i2, 3), 1)[0]
    else:
        return i2


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
        depart_time = [uniform(60 * 9, 60 * 18)]
        individual += customer_sequence + charging_sequence + depart_time
    return individual


def fitness(individual: IndividualType, fleet: Fleet, indices: IndicesType, starting_points: StartingPointsType,
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
    costs = np.array(fleet.cost_function())

    # Check if the solution is feasible
    feasible, penalization = fleet.feasible()

    # penalization
    if not feasible:
        penalization += penalization_constant

    fit = np.dot(costs, np.asarray(weights)) + penalization

    return fit, feasible


# THE ALGORITHM
def optimal_route_assignation(fleet: Fleet, hp: HyperParameters, save_to: str = None, best_ind=None):
    # TOOLBOX
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin, feasible=False)

    toolbox = base.Toolbox()
    toolbox.register("individual", random_individual, starting_points=hp.starting_points,
                     customers_to_visit=hp.customers_to_visit, charging_stations=fleet.network.charging_stations,
                     allowed_charging_operations=hp.r)
    toolbox.register("evaluate", fitness, fleet=fleet, indices=hp.indices, starting_points=hp.starting_points,
                     weights=hp.weights, penalization_constant=hp.penalization_constant,
                     allowed_charging_operations=hp.r)
    toolbox.register("mate", crossover, indices=hp.indices, index=None)
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

    # These will save statistics
    opt_data = GenerationsData([], [], [], [], [], [], fleet, hp, bestOfAll, bestOfAll.feasible)

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
        opt_data.save_opt_data(save_to)
    return routes, fleet, bestOfAll, feasible, toolbox, opt_data
