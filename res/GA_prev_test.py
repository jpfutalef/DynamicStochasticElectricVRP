import numpy as np
from random import randint
from random import uniform
from random import sample
from deap.tools import initCycle

from res.EV_utilities import feasible
from res.EV_utilities import distance
from res.ElectricVehicle import ElectricVehicle


def decode(individual, vehicles, allowed_charging_operations=2):
    """
    Decodes an individual to the corresponding node sequence and charging sequence. The individual has
    the following structure: ind = [customers, charg_ops, x0, ...]. S is the node sequence with the
    following structure  S = [S1, ..., Sm], where Si = [Si(0),..., Si(s_0-1)].  The L structure is the same as S
    structure. The x0 structure is x0=[x0_1,...x0_m].
    :param allowed_charging_operations: number of charging operations each vehicle is allowed to perform
    :param individual: The coded individual
    :param vehicles: a dictionary containing all vehicles instances
    :return: A 3-size tuple (S, L, x0)
    """
    # print("individual to decode: ", individual)
    S = {}
    L = {}
    x0 = {}
    i0 = 0
    i1 = 0

    for id_vehicle, vehicle in vehicles.items():
        vehicle: ElectricVehicle

        ni = vehicle.ni
        i1 += ni

        # iterate through every charging operation
        indices = [i1 + 3 * k for k in range(0, allowed_charging_operations)]
        charge_after = {individual[i]: (individual[i + 1], individual[i + 2]) for i in indices if
                        individual[i] != -1}

        nodeSequence = [0] * (ni + len(charge_after))
        chargingSequence = [0] * (ni + len(charge_after))

        # case: there are not recharging operations
        if len(charge_after) == 0:
            nodeSequence = individual[i0:i1]

        # case: there are recharging operations
        else:
            key = individual[i0]
            j = 0
            insertCS = False

            for i, _ in enumerate(nodeSequence):
                if insertCS:
                    nodeSequence[i] = charge_after[key][0]
                    chargingSequence[i] = charge_after[key][1]
                    insertCS = False

                elif individual[i0 + j] in charge_after.keys():
                    nodeSequence[i] = individual[i0 + j]
                    key = individual[i0 + j]
                    j += 1
                    insertCS = True

                else:
                    nodeSequence[i] = individual[i0 + j]
                    j += 1
        # Departure time
        x0_vehicle = individual[i1 + 3 * allowed_charging_operations]

        # Store in dictionary
        S[id_vehicle] = [0] + nodeSequence + [0]
        L[id_vehicle] = [0] + chargingSequence + [0]
        x0[id_vehicle] = x0_vehicle

        i0 += ni + 3 * allowed_charging_operations + 1
        i1 += 3 * allowed_charging_operations + 1

    return S, L, x0


def fitness(individual, vehicles, allowed_charging_operations=2, x2_0=80.0):
    """
    Calculates fitness of individual.
    :param individual: The individual to decode
    :param vehicles: dictionary with vehicle instances. They must have been assigned to customers
    :param allowed_charging_operations: maximum charging operations per ev
    :param x2_0: SOC of EVs leaving depot
    :return: the fitness of the individual
    """

    # Decode
    S, L, x0 = decode(individual, vehicles, allowed_charging_operations=allowed_charging_operations)

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

        # Ensure reset
        # vehicle.reset()

        # Sequences
        Si = S[id_vehicle]
        Li = L[id_vehicle]
        x1_0 = x0[id_vehicle]

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

    return travel_time_cost, charging_time_cost, energy_consumption_cost, charging_cost


def mutate(individual, vehicles, indices, allowed_charging_operations=2, index=None):
    """
    Mutataes individual.
    :param individual: the individual
    :param vehicles: dictionary containing vehicles instances by id
    :param indices: indices of sub blocks created by doing i0, i1, i2 = createImportantIndices(vehicles,
    allowed_charging_operations)
    :param allowed_charging_operations:
    :param index: If given, where to mutate
    :return: mutated individual
    """

    # print("Original individual: ", individual)
    # Lists with indices of sub blocks
    i0List, i1List, i2List = indices[0], indices[1], indices[2]

    # Choose a random index if not passed
    if index is None:
        index = randint(0, len(individual))

    # Find concerning block
    i = 0
    i0 = 0
    i1 = 0
    i2 = 0

    for i, (i0, i2) in enumerate(zip(i0List, i2List)):  # i is the EV id
        if i0 <= index <= i2:
            i1 = i1List[i]
            break

    # Case customer
    # print('Original Individual:', individual)
    # print('Index: ', index)
    if i0 <= index < i1:
        # print("Case customer", (i0, i1))
        case = "customer"
        i = randint(i0, i1 - 1)
        while True:
            j = randint(i0, i1 - 1)
            if j != i:
                break
        swapList(individual, i, j)

    # Case CS
    elif i1 <= index < i2:
        # print("Case charge station", (i1, i2))
        case = "CS"
        # FIXME make this more efficient. Might be good to return tuples with the
        #  values in the function that creates these

        # Find corresponding operation index
        j = 0
        csBlocks = [i1 + 3 * x for x in range(0, allowed_charging_operations)]
        for j in csBlocks:
            if j <= index <= j + 2:
                break

        # Choose if making a charging operation
        if randint(0, 1):
            # print("Charge....")
            # Choose a customer node randomly
            g = 0
            while g < 5000:
                customer = sample(vehicles[i].customers_to_visit, 1)[0]
                # print("After customer: ", individual[j])
                g += 1
                if customer not in [individual[x] for x in csBlocks]:
                    break

            individual[j] = customer
        else:
            # print("Don't charge")
            individual[j] = -1

        # Choose a random CS
        individual[j + 1] = sample(vehicles[i].network.ids_charge_stations, 1)[0]
        # print("At CS: ", individual[j+1])

        # Choose amount
        individual[j + 2] = uniform(0.0, 90.0)

        # print("The amount of: ", individual[j+2])

    # Case x0
    else:
        case = "x0"
        # print("Case x0", i2)
        individual[i2] += uniform(-60, 60)

    # print("Mutated individual: ", individual)
    # print("Case: ", case)
    return individual


def crossover(ind1, ind2, vehicles, indices, allowed_charging_operations=2, index=None):
    """
    Crossover of two individuals
    :param ind1: First individual
    :param ind2: Second individual
    :param vehicles: dict with vehicles info
    :param indices: indices of sub blocks created by doing i0, i1, i2 = createImportantIndices(vehicles,
    allowed_charging_operations)
    :param allowed_charging_operations:
    :param index: If given, where to do crossover
    :return:
    """
    i0List, i1List, i2List = indices[0], indices[1], indices[2]

    # Choose a random index if not passed
    if index is None:
        index = randint(0, len(ind1))

    # Find concerning block
    i = 0
    i0 = 0
    i1 = 0
    i2 = 0

    for i, (i0, i2) in enumerate(zip(i0List, i2List)):
        if i0 <= index <= i2:
            i1 = i1List[i]
            break

    # Case customer
    if i0 <= index < i1:
        swapBlock(ind1, ind2, i0, i1)

    # Case CS
    elif i1 <= index < i2:
        swapBlock(ind1, ind2, i1, i2)

    # Case x0
    else:
        swapBlock(ind1, ind2, i2, i2 + 1)

    return ind1, ind2


def swapList(l, i, j):
    """
    This function allows to swap two elements in a list, given two positions.
    :param l: the list
    :param i: first position
    :param j: second position
    """
    l[i], l[j] = l[j], l[i]


def swapBlock(l1, l2, i, j):
    """
    This function allows to swap block within a list, given two positions.
    :param l1: first list
    :param l2: second list
    :param i: first position
    :param j: second position
    """
    l1[i: j], l2[i:j] = l2[i:j], l1[i:j]


def createImportantIndices(vehicles, allowed_charging_operations=2):
    """
    Creates the indices of sub blocks.
    :param vehicles: dict with vehicles info by id
    :param allowed_charging_operations:
    :return: tuple with indices list (i0, i1, i2)
    """
    i0List = []
    i1List = []
    i2List = []

    i0 = 0
    i1 = 0
    i2 = 0

    for id_vehicle, vehicle in vehicles.items():
        vehicle: ElectricVehicle
        
        ni = vehicle.ni
        i1 += ni
        i2 += ni + 3 * allowed_charging_operations

        i0List.append(i0)
        i1List.append(i1)
        i2List.append(i2)

        i0 += ni + 3 * allowed_charging_operations + 1
        i1 += 3 * allowed_charging_operations + 1
        i2 += 1

    return i0List, i1List, i2List


def createRandomIndividual(vehicles, allowed_charging_operations=2):
    """
    Creates a random individual
    :param vehicles: dict with vechiles info
    :param allowed_charging_operations:
    :return: a random individual
    """
    individual = []
    for id_vehicle, vehicle in vehicles.items():
        vehicle: ElectricVehicle

        customerSequence = sample(vehicle.customers_to_visit, vehicle.ni)
        #seq = [lambda: -1, lambda: 0, lambda: 10.0]
        seq = [lambda: sample(customerSequence, 1), lambda: sample(vehicle.network.ids_charge_stations, 1),
               lambda: uniform(0.0, 90.0)]

        chargingSequence = initCycle(list, seq, n=allowed_charging_operations)
        departingTime = [randint(0, 24 * 60)]
        individual += customerSequence + chargingSequence + departingTime

    return individual


def feasibleIndividual(individual, vehicleDict, allowed_charging_operations=2):
    nodeSeq, charSeq, x0Seq = decode(individual, vehicleDict,
                                     allowed_charging_operations=allowed_charging_operations)
    return feasible(nodeSeq, charSeq, x0Seq, vehicleDict)


def distanceToFeasibleZone(individual, vehicleDict, allowed_charging_operations=2):
    nodeSeq, charSeq, x0Seq = decode(individual, vehicleDict,
                                     allowed_charging_operations=allowed_charging_operations)
    return distance(nodeSeq, charSeq, x0Seq, vehicleDict)
