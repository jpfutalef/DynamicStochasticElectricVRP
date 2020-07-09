from random import randint, uniform, sample, random

from GATools import *

# CLASSES


# TYPES
IndividualType = List
IndicesType = Dict[int, Tuple[int, int, int]]
StartingPointsType = Dict[int, InitialCondition]
RouteVector = Tuple[Tuple[int, ...], Tuple[float, ...]]
RouteDict = Dict[int, Tuple[RouteVector, float, float, float]]


# FUNCTIONS


def decode(individual: IndividualType, indices: IndicesType, init_state: StartingPointsType, r: int) -> RouteDict:
    routes = {}
    for id_ev, (i0, i1, i2) in indices.items():
        Sk = individual[i0:i1]
        Lk = [0] * len(Sk)
        chg_block = individual[i1:i2]
        chg_ops = [(chg_block[3*i], chg_block[3*i+1], chg_block[3*i+2]) for i in range(r)]
        for customer, charging_station, amount in chg_ops:
            if customer != -1:
                i = Sk.index(customer)
                Sk = Sk[:i+1] + [charging_station] + Sk[i+1:]
                Lk = Lk[:i+1] + [amount] + Lk[i+1:]
        Sk = tuple([0] + Sk + [0])
        Lk = tuple([0] + Lk + [0])
        x0 = individual[i2]
        routes[id_ev] = ((Sk, Lk), x0, init_state[id_ev].x2_0, init_state[id_ev].x3_0)
    return routes


def mutate(individual: IndividualType, indices: IndicesType, charging_stations: Tuple,
           block_probability: Tuple[float, float, float], repeat=1) -> None:
    for i in range(repeat):
        index = random_block_index(indices, block_probability)
        for id_ev, (i0, i1, i2) in indices.items():
            if i0 <= index <= i2:
                # Case customer
                if i0 <= index < i1:
                    case = random()
                    if case <= 0.85:
                        i, j = randint(i0, i1 - 1), randint(i0, i1 - 1)
                        while i == j:
                            j = randint(i0, i1 - 1)
                        swap_elements(individual, i, j)
                    elif case < 0.95:
                        i = randint(i0, i1 - 1)
                        individual[i0:i1] = individual[i:i1] + individual[i0:i]
                    else:
                        individual[i0:i1] = sample(individual[i0:i1], i1 - i0)

                # Case CS
                elif i1 <= index < i2:
                    i = i1 + 3 * int((index - i1) / 3)
                    if randint(0, 1):
                        individual[i] = sample(individual[i0:i1], 1)[0]
                    else:
                        individual[i] = -1
                    individual[i + 1] = sample(charging_stations, 1)[0]
                    new_val = abs(individual[i + 2] + uniform(-10, 10))
                    new_val = new_val if new_val <= 90 else 90
                    individual[i + 2] = new_val
                    #individual[i + 2] = abs(individual[i + 1] + uniform(-10, 10))

                # Case depart time
                else:
                    amount = uniform(-20.0, 20.0)
                    individual[i2] = abs(individual[i2] + amount)
                break


def crossover(ind1: IndividualType, ind2: IndividualType, indices: IndicesType,
              block_probability: Tuple[float, float, float], repeat=1) -> None:
    for i in range(repeat):
        index = random_block_index(indices, block_probability)
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


def swap_elements(l, i, j):
    l[i], l[j] = l[j], l[i]


def swap_block(l1, l2, i, j):
    l1[i: j], l2[i:j] = l2[i:j], l1[i:j]


def random_block_index(indices: IndicesType, block_probability: Tuple[float, float, float]) -> int:
    id_ev = sample(indices.keys(), 1)[0]
    i0, i1, i2 = indices[id_ev]
    if random() < block_probability[0]:
        return randint(i0, i1 - 1)
    elif random() < block_probability[1]:
        return sample(range(i1, i2, 3), 1)[0]
    else:
        return i2


def block_indices(customers_to_visit: Dict[int, Tuple[int, ...]], r:int) -> IndicesType:
    indices = {}
    i0, i1, i2 = 0, 0, 0
    for id_vehicle, customers in customers_to_visit.items():
        ni = len(customers)
        i1 += ni
        i2 += ni + 3 * r

        indices[id_vehicle] = (i0, i1, i2)

        i0 += ni + 3 * r + 1
        i1 += 3 * r + 1
        i2 += 1

    return indices


def random_individual(customers_per_vehicle: Dict[int, Tuple[int, ...]], charging_stations: Tuple[int, ...],
                      r: int) -> IndividualType:
    individual = []
    for id_ev, customers in customers_per_vehicle.items():
        customer_sequence = sample(customers, len(customers))
        charging_operations = []
        for _ in range(r):
            charging_operations.append(sample(customer_sequence + [-1]*r, 1)[0])
            charging_operations.append(sample(charging_stations, 1)[0])
            charging_operations.append(uniform(0, 90))
        depart_time = [uniform(60 * 9, 60 * 18)]
        individual += customer_sequence + charging_operations + depart_time
    return individual


def fitness(individual: IndividualType, fleet: Fleet, indices: IndicesType, init_state: StartingPointsType, r: int,
            weights=(1.0, 1.0, 1.0, 1.0), penalization_constant=500000):
    routes = decode(individual, indices, init_state, r)
    fleet.set_routes_of_vehicles(routes)
    fleet.create_optimization_vector()
    cost_tt, cost_ec, cost_chg_op, cost_chg_cost = fleet.cost_function()
    feasible, penalization = fleet.feasible()

    if not feasible:
        penalization += penalization_constant

    costs = np.array([cost_tt, cost_ec, cost_chg_op, cost_chg_cost])
    fit = np.dot(costs, np.asarray(weights)) + penalization
    return fit, feasible
