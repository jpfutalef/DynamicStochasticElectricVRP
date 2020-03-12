import numpy as np
from random import randint
from random import uniform
from random import sample
from deap.tools import initCycle

from res.EV_utilities import feasible
from res.EV_utilities import distance
from res.ElectricVehicle import ElectricVehicle, feasible, createOptimizationVector


def decode(individual, indices, starting_points, allowed_charging_operations=2):
    """
    Decodes an individual to the corresponding node sequence and charging sequence. S is the node sequence with the
    following structure  S = [S1, ..., Sm], where Si = [Si(0),..., Si(s_0-1)].  The L structure is the same as S
    structure.
    :param allowed_charging_operations: number of charging operations each vehicle is allowed to perform
    :param individual: The coded individual
    :param indices: a
    :param starting_points: dictionary with information about the start of sequences {id_vehicle:(S0, L0, x1_0,
    x2_0),..., }
    :return: A 3-size tuple (S, L, x0)
    """
    # print("individual to decode: ", individual)
    S = {}
    L = {}

    for id_ev, (i0, i1) in enumerate(indices):  # i is the EV id
        charging_operations = [(individual[i1 + 3 * i], individual[i1 + 3 * i + 1], individual[i1 + 3 * i + 2])
                               for i in range(allowed_charging_operations) if individual[i1 + 3 * i] != -1]
        customers_block = individual[i0:i1]
        ni = len(customers_block)

        init_node_seq = [starting_points[id_ev][0]] + customers_block
        init_charging_seq = [starting_points[id_ev][1]] + [0] * ni

        # case: there are not recharging operations
        if len(charging_operations) == 0:
            node_sequence = init_node_seq
            charging_sequence = init_charging_seq

        # case: there are recharging operations
        else:
            node_sequence = [0] * (ni + len(charging_operations) + 1)
            charging_sequence = [0] * (ni + len(charging_operations) + 1)

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
        S[id_ev] = node_sequence + [0]
        L[id_ev] = charging_sequence + [0]

    return S, L


def mutate(individual, indices, starting_points, customers, charging_stations, allowed_charging_operations=2,
           index=None):
    # Choose a random index if not passed
    if index is None:
        index = randint(0, len(individual))

    # Find concerning block
    i = 0
    i0, i1 = indices[0]
    for i, (i0, i1) in enumerate(indices):  # i is the EV id
        if i0 <= index <= i1 + 3 * allowed_charging_operations - 1:
            break

    # Case customer
    if i0 <= index < i1:
        i = randint(i0, i1 - 1)
        while True:
            j = randint(i0, i1 - 1)
            if j != i:
                break
        swap_elements(individual, i, j)

    # Case CS
    elif i1 <= index <= i1 + 3 * allowed_charging_operations - 1:
        # Find corresponding operation index
        j = 0
        for j in range(allowed_charging_operations):
            if i1 + j * allowed_charging_operations <= index <= i1 + j * allowed_charging_operations + 2:
                break

        # Choose if making a charging operation
        if randint(0, 1):
            # Choose a customer node randomly
            while True:
                sample_space = [starting_points[i][0]] + customers[i]
                customer = sample(sample_space, 1)[0]
                # Ensure customer is not already chosen
                if customer not in [individual[i1 + 3 * x] for x in range(allowed_charging_operations)]:
                    break
            individual[i1 + 3 * j] = customer

        else:
            individual[i1 + 3 * j] = -1

        # Choose a random CS anyways
        individual[i1 + 3 * j + 1] = sample(charging_stations, 1)[0]

        # Choose amount anyways
        individual[i1 + 3 * j + 2] = uniform(0.0, 90.0)
    return individual


def crossover(ind1, ind2, indices, allowed_charging_operations=2, index=None):
    # Choose a random index if not passed
    if index is None:
        index = randint(0, len(ind1))

    # Find concerning block
    i0, i1 = indices[0]
    for i, (i0, i1) in enumerate(indices):  # i is the EV id
        if i0 <= index <= i1 + 3 * allowed_charging_operations - 1:
            break

    # Case customer
    if i0 <= index < i1:
        swap_block(ind1, ind2, i0, i1)

    # Case CS
    elif i1 <= index <= i1 + 3 * allowed_charging_operations - 1:
        swap_block(ind1, ind2, i1, i1 + 3 * allowed_charging_operations - 1)


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
    :param i: first position
    :param j: second position
    """
    l1[i: j], l2[i:j] = l2[i:j], l1[i:j]


def block_indices(customer_count, allowed_charging_operations=2):
    """
    Creates the indices of sub blocks.
    :param customer_count: list with the amount of customer to visit, after initial condition
    :param allowed_charging_operations:
    :return: tuples list with indices [(i0, j0), (i1, j1),...] where iN represent location of the beginning of customers_per_vehicle
    block and jN location of the beginning of charging operations block of evN in the individual, respectively.
    """
    indices = []

    i0 = 0
    i1 = 0
    for ni in customer_count:
        i1 += ni

        indices.append((i0, i1))

        i0 += ni + 3 * allowed_charging_operations
        i1 += 3 * allowed_charging_operations

    return indices


def random_individual(indices, starting_points, customers_per_vehicle, charging_stations,
                      allowed_charging_operations=2):
    """
    Creates a random individual
    :param allowed_charging_operations:
    :return: a random individual
    """
    individual = []
    for init_point, customers, (i0, i1) in zip(starting_points.values(), customers_per_vehicle, indices):
        customer_sequence = sample(customers, len(customers))

        sample_space = customers + [init_point[0]] + [-1] * allowed_charging_operations
        after_customers = sample(sample_space, allowed_charging_operations)
        charging_sequence = [0] * allowed_charging_operations * 3
        for i, customer in enumerate(after_customers):
            charging_sequence[3*i] = customer
            charging_sequence[3*i + 1] = sample(charging_stations, 1)[0]
            charging_sequence[3*i + 2] = uniform(0.0, 90.0)

        individual += customer_sequence + charging_sequence

    return individual


def fitness(individual, vehicles, indices, starting_points, weights=(1.0, 1.0, 1.0, 1.0), penalization_constant=500000,
            allowed_charging_operations=2):
    """
    Calculates fitness of individual.
    :param individual: The individual to decode
    :param vehicles: dictionary with vehicle instances. They must have been assigned to customers_per_vehicle
    :param starting_points: dictionary with info of the initial state {id_vehicle:(S0, L0, x1_0, x2_0)}
    :param allowed_charging_operations: maximum charging operations per ev
    :param weights: tuple with weights of each variable in cost function (w1, w2, w3, w4)
    :param penalization_constant: positive number that represents the penalization of unfeasible individual
    :return: the fitness of the individual
    """

    #TODO remove ElectricVehicle dependence
    # Decode
    S, L = decode(individual, indices, starting_points, allowed_charging_operations=allowed_charging_operations)

    # Lists to store costs
    travel_time_costs = []
    energy_consumption_costs = []
    charging_time_costs = []
    charging_costs = []

    append_travel_time = travel_time_costs.append
    append_energy_consumption = energy_consumption_costs.append
    append_charging_time = charging_time_costs.append
    append_charging_cost = charging_costs.append

    # Iterate each vehicle
    for id_vehicle, vehicle in vehicles.items():
        vehicle: ElectricVehicle

        # Sequences
        Si = S[id_vehicle]
        Li = L[id_vehicle]
        x1_0 = starting_points[id_vehicle][2]
        x2_0 = starting_points[id_vehicle][3]

        # Set and iterate state
        vehicle.set_sequences(Si, Li, x1_0, x2_0)
        vehicle.iterateState()

        # Store
        append_travel_time(vehicle.cost_travel_time())
        append_charging_time(vehicle.cost_charging_time())
        append_energy_consumption(vehicle.cost_energy_consumption())
        append_charging_cost(vehicle.cost_energy_consumption())

    # Obtain sum of costs
    travel_time_cost = np.sum(travel_time_costs)
    charging_time_cost = np.sum(charging_time_costs)
    energy_consumption_cost = np.sum(energy_consumption_costs)
    charging_cost = np.sum(charging_costs)

    op_vector = createOptimizationVector(vehicles)
    is_feasible, penalization = feasible(op_vector, vehicles)

    if not is_feasible:
        penalization += penalization_constant

    costs = np.array([travel_time_cost, charging_time_cost, energy_consumption_cost, charging_cost])
    fit = np.dot(costs, np.array(weights)) + penalization

    return fit,


if __name__ == '__main__':
    customers_to_visit = [[1, 4, 5],
                          [2, 3, 6]]
    customer_count = [len(x) for x in customers_to_visit]

    charging_stations = [7, 8]
    all_ch_ops = 2

    # Indices
    indices = block_indices(customer_count, allowed_charging_operations=all_ch_ops)

    # Starting points (S0, L0, x1_0, x2_0)
    init_state = {0: (0, 0, 150., 80.), 1: (0, 0, 250., 80.)}

    ind1 = [5, 1, 4, -1, 8, 10.5, 1, 7, 15.5,
            6, 3, 2, 6, 8, 11.5, -1, 7, 12.5]
    ind2 = [5, 1, 4, 5, 8, 10.5, 1, 7, 15.5,
            6, 3, 2, 6, 8, 11.5, 3, 7, 12.5]
    ind3 = [5, 1, 4, 0, 8, 10.5, 1, 7, 15.5,
            6, 3, 2, 6, 8, 11.5, 3, 7, 12.5]

    # Decode i1
    S, L = decode(ind1, indices, init_state, allowed_charging_operations=all_ch_ops)
    print(S, L)

    # Decode i2
    S, L = decode(ind2, indices, init_state, allowed_charging_operations=all_ch_ops)
    print(S, L)

    # Decode i3
    S, L = decode(ind3, indices, init_state, allowed_charging_operations=all_ch_ops)
    print(S, L)

    # Mate i1 and i2
    print('Individual 1:', ind1)
    print('Individual 2:', ind2, '\nMate...')
    while True:
        index = input()
        if index == '':
            break
        else:
            index = int(index)
        crossover(ind1, ind2, indices, allowed_charging_operations=all_ch_ops, index=index)
        print('Individual 1:', ind1)
        print('Individual 2:', ind2)

    # Mutate i3
    print('Individual 3:', ind3, '\nMutate...')
    while True:
        index = input()
        if index == '':
            break
        else:
            index = int(index)
        mutate(ind3, indices, init_state, customers_to_visit, charging_stations,
               allowed_charging_operations=all_ch_ops, index=index)
        print('Individual 3:', ind3)

    # Create random individuals
    print('Random individuals')
    while True:
        index = input()
        if index == 's':
            break
        else:
            print(random_individual(indices, init_state, customers_to_visit, charging_stations,
                                    allowed_charging_operations=all_ch_ops))
