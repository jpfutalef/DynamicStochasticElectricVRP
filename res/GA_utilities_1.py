import numpy as np
from random import randint
from random import uniform
from random import sample
from deap.tools import initCycle

from res.EV_utilities import feasible
from res.EV_utilities import distance


def decodeFunction(individual, vehiclesDict, allowed_charging_operations=2):
    """
    Decodes an individual to the corresponding node sequence, charging sequence and starting time. S is the node
    sequence with the following structure  S = [S1, ..., Sm], where Si = [Si(0),..., Si(s_0-1)].  The L structure
    is the same as S structure. The x0 structure is x0=[x0_1,...x0_m].
    :param allowed_charging_operations: number of charging operations each vehicle is allowed to perform
    :param individual: The coded individual
    :param vehiclesDict: a dictionary containing all vehicles instances
    :return: A 3-size tuple (S, L, x0)
    """
    #print("individual to decode: ", individual)
    S = {}
    L = {}
    x0 = {}
    i0 = 0
    i1 = 0

    # TODO the following assumes dict keys are ordered
    for k, v in vehiclesDict.items():
        ni = v.customerCount
        i1 += ni

        # iterate through every charging operation
        indices = [i1 + 3 * k for k in range(0, allowed_charging_operations)]
        chargeAfterDict = {individual[i]: (individual[i + 1], individual[i + 2]) for i in indices if individual[i] != -1}

        nodeSequence = [0] * (ni + len(chargeAfterDict))
        chargingSequence = [0] * (ni + len(chargeAfterDict))

        # case: there are not recharging operations
        if len(chargeAfterDict) == 0:
            nodeSequence = individual[i0:i1]

        # case: there are recharging operations
        else:
            key = individual[i0]
            j = 0
            insertCS = False

            for i, _ in enumerate(nodeSequence):
                if insertCS:
                    nodeSequence[i] = chargeAfterDict[key][0]
                    chargingSequence[i] = chargeAfterDict[key][1]
                    insertCS = False

                elif individual[i0 + j] in chargeAfterDict.keys():
                    nodeSequence[i] = individual[i0 + j]
                    key = individual[i0 + j]
                    j += 1
                    insertCS = True

                else:
                    nodeSequence[i] = individual[i0 + j]
                    j += 1

        x0_vehicle = individual[i1 + 3 * allowed_charging_operations]

        # Store in dictionary
        # TODO the following assumes EV start and end at depot. Find a way to change this.
        S[k] = [0] + nodeSequence + [0]
        L[k] = [0] + chargingSequence + [0]
        x0[k] = x0_vehicle

        v.updateSequences(S[k], L[k], x0[k])

        # print("Vehicle", v.id, "sequences: ")
        # print("node sequence: ", S[k])
        # print("charging sequence: ", L[k])
        # print("x0:", x0[k], "\n")

        i0 += ni + 3 * allowed_charging_operations + 1
        i1 += 3 * allowed_charging_operations + 1
    return S, L, x0


def fitness(individual, vehiclesDict, allowed_charging_operations=2):
    # ensure reset
    for k, v in vehiclesDict.items():
        v.travelTimeCost = 0
        v.chargingTimeCost = 0

    # Decode
    decodeFunction(individual, vehiclesDict, allowed_charging_operations=allowed_charging_operations)

    # Zero disturbance by the moment
    stateSequences = {}
    for vehicleId in vehiclesDict.keys():
        seqEta = np.zeros(vehiclesDict[vehicleId].si)
        stateSequences[vehicleId] = vehiclesDict[vehicleId].createStateSequenceStatic(seqEta)

    # Cost function of sequences
    travelTimeCost = np.sum([vehiclesDict[x].travelTimeCost for x in vehiclesDict.keys()])
    chargingTimeCost = np.sum([vehiclesDict[x].chargingTimeCost for x in vehiclesDict.keys()])
    totalCost = travelTimeCost + chargingTimeCost

    # reset
    for k, v in vehiclesDict.items():
        v.travelTimeCost = 0
        v.chargingTimeCost = 0

    return -totalCost,


def mutate(individual, vehiclesDict, allowed_charging_operations=2, index=None):
    # print("Original individual: ", individual)
    # indices lists TODO: prevent creation of these list every time
    i0List, i1List, i2List = createImportantIndices(vehiclesDict,
                                                    allowed_charging_operations=allowed_charging_operations)

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
                customer = sample(vehiclesDict[i].customersId, 1)[0]
                # print("After customer: ", individual[j])
                g += 1
                if customer not in [individual[x] for x in csBlocks]:
                    break

            individual[j] = customer
        else:
            # print("Don't charge")
            individual[j] = -1

        # Choose a random CS
        individual[j + 1] = sample(vehiclesDict[i].networkInfo['CS_LIST'], 1)[0].id
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


def crossover(ind1, ind2, vehiclesDict, allowed_charging_operations=2, index=None):
    i0List, i1List, i2List = createImportantIndices(vehiclesDict,
                                                    allowed_charging_operations=allowed_charging_operations)

    # Choose a random index if not passed
    if index is None:
        index = randint(0, len(ind1))

    # Find concerning block
    i = 0  # vehicle ID
    i0 = 0
    i1 = 0
    i2 = 0

    for i, (i0, i2) in enumerate(zip(i0List, i2List)):  # i is the EV id
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


def createImportantIndices(vehiclesDict, allowed_charging_operations=2):
    # TODO docu
    # Create indices
    i0List = []
    i1List = []
    i2List = []

    i0 = 0
    i1 = 0
    i2 = 0

    # TODO the following assumes dict keys are ordered
    for k, v in vehiclesDict.items():
        ni = v.customerCount
        i1 += ni
        i2 += ni + 3 * allowed_charging_operations

        i0List.append(i0)
        i1List.append(i1)
        i2List.append(i2)

        i0 += ni + 3 * allowed_charging_operations + 1
        i1 += 3 * allowed_charging_operations + 1
        i2 += 1

    return i0List, i1List, i2List


def createRandomIndividual(vehiclesDict, allowed_charging_operations=2):
    individual = []
    for k, v in vehiclesDict.items():
        customerSequence = sample(v.customersId, len(v.customersId))
        seq = [lambda: -1, lambda: 0, lambda: 10.0]
        chargingSequence = initCycle(list, seq, n=allowed_charging_operations)
        departingTime = [randint(0, 24 * 60)]
        individual += customerSequence + chargingSequence + departingTime

    return individual


def feasibleIndividual(individual, vehicleDict, allowed_charging_operations=2):
    nodeSeq, charSeq, x0Seq = decodeFunction(individual, vehicleDict,
                                             allowed_charging_operations=allowed_charging_operations)
    return feasible(nodeSeq, charSeq, x0Seq, vehicleDict)


def distanceToFeasibleZone(individual, vehicleDict, allowed_charging_operations=2):
    nodeSeq, charSeq, x0Seq = decodeFunction(individual, vehicleDict,
                                             allowed_charging_operations=allowed_charging_operations)
    return distance(nodeSeq, charSeq, x0Seq, vehicleDict)
