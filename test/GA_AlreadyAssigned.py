from res.GA_AlreadyAssigned import *
from models.Fleet import from_xml, InitialCondition

if __name__ == '__main__':
    #### IMPORT FLEET FROM FILE ####
    print('***** FROM FILE *****')
    path = '../data/GA_implementation_xml/test_instance/test_instance_realtime.xml'
    fleet = from_xml(path, assign_customers=True)
    customers_to_visit = {ev_id: ev.assigned_customers for ev_id, ev in fleet.vehicles.items()}
    indices = block_indices(customers_to_visit, allowed_charging_operations=2)

    # Create random individuals using file
    starting_points = {ev_id: InitialCondition(0, 0, 0, 80., sum([fleet.network.nodes[x].demand for x in ev.assigned_customers])) for ev_id, ev in fleet.vehicles.items()}
    random_ind1 = random_individual(indices, starting_points, customers_to_visit, fleet.network.charging_stations, 2)
    random_ind2 = random_individual(indices, starting_points, customers_to_visit, fleet.network.charging_stations, 2)
    random_ind3 = random_individual(indices, starting_points, customers_to_visit, fleet.network.charging_stations, 2)

    # Their fitness
    fitness1, feasible1 = fitness(random_ind1, fleet, indices, starting_points)
    fitness2, feasible2 = fitness(random_ind2, fleet, indices, starting_points)
    fitness3, feasible3 = fitness(random_ind3, fleet, indices, starting_points)

    print(f'Fitness 1: {fitness1} (feasible: {feasible1})')
    print(f'Fitness 2: {fitness2} (feasible: {feasible2})')
    print(f'Fitness 3: {fitness3} (feasible: {feasible3})')
    a = 1
