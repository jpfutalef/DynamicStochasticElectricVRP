from res.GA_Assignation import *
from res.GATools import *
from res.GA_AlreadyAssigned2 import optimal_route_assignation as improve_route
from res.GA_AlreadyAssigned2 import individual_from_routes

from os import listdir
from os.path import isfile, join

# %% 1. Specify instance location and capacities to iterate
data_folder = 'data/instances/'
instances = [f for f in listdir('data/instances/') if isfile(join('data/instances/', f))]
#instances = ['c75cs14_20x20km.xml']

capacities = [4, 3, 2, 1]
soc_policy = (20, 95)

# %% 2. Instances iteration
for instance in instances:
    instance_filename = instance
    path = f'{data_folder}{instance_filename}'
    print(f'Opening:\n {path}')
    for cap in capacities:
        fleet = from_xml(path, assign_customers=False)
        '''
        fleet.network.draw(save_to=None, width=0.02,
                           edge_color='grey', markeredgecolor='black',
                           markeredgewidth=2.0)
        '''
        # %% 3. Modify CS capacities and set initial fleet size
        weight_sum = sum([fleet.network.demand(i) for i in fleet.network.customers])
        max_weight = fleet.vehicles[0].max_payload
        init_fleet_size = int(weight_sum/ max_weight) + 1
        fleet.resize_fleet(init_fleet_size)
        fleet.modify_cs_capacities(cap)
        fleet.new_soc_policy(soc_policy[0], soc_policy[1])

        # %% 4. GA hyper-parameters
        CXPB, MUTPB = 0.7, 0.9
        num_individuals = int(len(fleet.network)*1.5) + int(len(fleet)*10) + 50
        max_generations = num_individuals*3
        penalization_constant = 500000
        weights = (.2, 1.2, 20., 20., .3)  # cost_tt, cost_ec, cost_chg_op, cost_chg_cost, cost_wait_time
        keep_best = 1  # Keep the 'keep_best' best individuals
        tournament_size = 3
        r = 4
        hyper_parameters = HyperParameters(num_individuals, max_generations, CXPB, MUTPB,
                                           tournament_size=tournament_size,
                                           penalization_constant=penalization_constant,
                                           keep_best=keep_best,
                                           weights=weights,
                                           r=r)

        CXPB, MUTPB = 0.7, 0.9
        num_individuals = int(len(fleet.network) * 1.5) + int(len(fleet) * 10) + 50
        max_generations = num_individuals * 2
        penalization_constant = 500000
        weights = (.2, 1.2, 20., 20., .3)  # cost_tt, cost_ec, cost_chg_op, cost_chg_cost, cost_wait_time
        keep_best = 1  # Keep the 'keep_best' best individuals
        tournament_size = 3
        crossover_repeat = 1
        mutation_repeat = 2
        hyper_parameters_improve = HyperParameters(num_individuals, max_generations, CXPB, MUTPB,
                                                   tournament_size=tournament_size,
                                                   penalization_constant=penalization_constant,
                                                   keep_best=keep_best,
                                                   weights=weights,
                                                   crossover_repeat=crossover_repeat,
                                                   mutation_repeat=mutation_repeat)

        # %% 5. Specify data folder
        instance_name = instance[:-4]
        try:
            os.mkdir(f'{data_folder}{instance_name}/')
        except FileExistsError:
            pass
        save_to = f'{data_folder}{instance_name}/{cap}/'

        # %% 6. Run algorithm
        feasible1, feasible2 = False, False
        bestOfAll1 = None
        bestOfAll2 = None
        for k in range(6):
            # Iterate for init_fleet_size + k vehicles
            routes, fleet, bestOfAll1, feasible1, toolbox1, optData1 = optimal_route_assignation(fleet,
                                                                                                 hyper_parameters,
                                                                                                 save_to,
                                                                                                 best_ind=bestOfAll1,
                                                                                                 savefig=True,
                                                                                                 plot_best_generation=False)

            bestOfAll2 = individual_from_routes(routes, fleet)
            routes, fleet, bestOfAll2, feasible2, toolbox2, optData2 = improve_route(fleet, hyper_parameters_improve,
                                                                                     save_to, bestOfAll2, savefig=True)
            best_fitness, best_is_feasible = toolbox2.evaluate(bestOfAll2)
            if not feasible1 and not feasible2:
                # Not fasible
                print('INCREASING FLEET SIZE BY 1...')
                fleet.resize_fleet(len(fleet) + 1)
                pos = np.random.randint(len(fleet.network.customers) + len(fleet))
                bestOfAll1.insert(pos, '|')
                bestOfAll1.append(np.random.uniform(7*60, 10*60))
                for i in range(hyper_parameters.r):
                    chg_op = [-1, sample(fleet.network.charging_stations, 1)[0], uniform(10, 20)]
                    index = -len(fleet)
                    bestOfAll1 = bestOfAll1[:index] + chg_op + bestOfAll1[index:]
                hyper_parameters.num_individuals += 15
                hyper_parameters_improve.num_individuals += 10
                hyper_parameters.max_generations += 10
                hyper_parameters_improve.max_generations += 10
            else:
                # At least one is feasible
                break

        # %% 7. If both solutions are infeasible, then the instance is infeasible with this capacity and less.
        if not feasible1 and not feasible2:
            break
