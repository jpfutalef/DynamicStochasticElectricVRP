from res.GA_Online import *
from models.Fleet import from_xml

if __name__ == '__main__':
    # Test GA operations
    customers_to_visit = {0: (1, 4, 5),
                          1: (2, 3, 6)}

    charging_stations = (7, 8)
    all_ch_ops = 2

    # Indices
    indices = block_indices(customers_to_visit, allowed_charging_operations=all_ch_ops)

    # Starting points (S0, L0, x1_0, x2_0, x3_0)
    init_state = {0: InitialCondition(0, 0, 150., 80., 0.8),
                  1: InitialCondition(0, 0, 250., 80., 0.9)}

    ind1 = [5, 1, 4, -1, 8, 10.5, 1, 7, 15.5,
            6, 3, 2, 6, 8, 11.5, -1, 7, 12.5]
    ind2 = [1, 4, 5, 5, 8, 12.5, 4, 7, 13.5,
            6, 2, 3, 2, 7, 11.5, 3, 8, 12.5]
    ind3 = [5, 1, 4, 0, 8, 10.5, 1, 7, 15.5,
            6, 3, 2, 6, 8, 11.5, 3, 7, 12.5]

    # Decode i1
    route1 = decode(ind1, indices, init_state, allowed_charging_operations=all_ch_ops)
    print(ind1, '\n', route1)

    # Decode i2
    route2 = decode(ind2, indices, init_state, allowed_charging_operations=all_ch_ops)
    print(ind2, '\n', route2)

    # Decode i3
    route3 = decode(ind3, indices, init_state, allowed_charging_operations=all_ch_ops)
    print(ind3, '\n', route3)

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

    #### IMPORT FLEET FROM FILE ####
    print('***** FROM FILE *****')
    path = '../data/GA_implementation_xml/test_instance/test_instance_realtime.xml'
    fleet = from_xml(path, assign_customers=True)
    customers_to_visit = {ev_id: ev.assigned_customers for ev_id, ev in fleet.vehicles.items()}
    starting_points = {ev_id: fleet.starting_points[ev_id] for ev_id, ev in fleet.vehicles.items()}
    ind = block_indices(customers_to_visit, allowed_charging_operations=2)

    # Create random individuals using file
    random_ind = random_individual(ind, starting_points, customers_to_visit, fleet.network.charging_stations,2)
    a = 1