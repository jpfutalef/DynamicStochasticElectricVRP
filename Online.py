import os
from os import listdir
from os.path import isfile, join

import numpy as np
from random import sample, uniform

from models.OnlineFleet import from_xml
from models.Network import from_xml as net_from_xml
from res.alphaGA import optimal_route_assignation
from res.betaGA import HyperParameters
from res.betaGA import optimal_route_assignation as improve_route

# %% 1. Specify instances location
folder = 'data/online/'
fleet_path = f'{folder}fleet21.xml'
network_path = f'{folder}network21.xml'
results_folder = f'{folder}instance21/'

# %% 2. CS capacities and SOC policy
cs_capacity = 3
soc_policy = (20, 95)

# %% Number of optimizations
opt_range = 1

# %% 3. Solve instances
for _ in range(opt_range):
    fleet = from_xml(fleet_path, assign_customers=False, with_routes=False, instance=False)
    net = net_from_xml(network_path, instance=False)
    fleet.set_network(net)

    fleet.modify_cs_capacities(cs_capacity)
    fleet.new_soc_policy(soc_policy[0], soc_policy[1])

    # %% 4. GA hyper-parameters
    num_individuals = int(len(fleet.network) * 1.5) + int(len(fleet) * 10) + 50
    K1 = 100. * len(fleet.network) + 1000 * len(fleet)
    hp_alpha = HyperParameters(num_individuals=num_individuals,
                               max_generations=num_individuals * 3,
                               CXPB=0.7,
                               MUTPB=0.9,
                               weights=(0.5 / 2.218, 1. / 0.4364, 1. / 8, 1. / 80, 1.2),
                               K1=K1,
                               K2=K1 * 2.5,
                               keep_best=1,
                               tournament_size=3,
                               r=4,
                               alpha_up=soc_policy[1],
                               algorithm_name='alphaGA')

    num_individuals = int(len(fleet.network) * 1.5) + int(len(fleet) * 10) + 50
    K1 = 100. * len(fleet.network) + 1000 * len(fleet)
    hp_beta = HyperParameters(num_individuals=num_individuals,
                              max_generations=num_individuals * 3,
                              CXPB=0.7,
                              MUTPB=0.9,
                              weights=(0.5 / 2.218, 1. / 0.4364, 1. / 8, 1. / 80, 1.2),
                              K1=K1,
                              K2=K1 * 2.5,
                              keep_best=1,
                              tournament_size=3,
                              r=4,
                              alpha_up=soc_policy[1],
                              algorithm_name='betaGA',
                              crossover_repeat=1,
                              mutation_repeat=1)

    # %% 5. Specify data folder
    # Main instance folder
    instance_folder = results_folder + '/'
    try:
        os.mkdir(instance_folder)
    except FileExistsError:
        pass

    # Main optimization folder
    opt_folders = [d for d in os.listdir(instance_folder) if os.path.isdir(os.path.join(instance_folder, d))]
    if opt_folders:
        opt_num = str(max([int(i[-1]) for i in opt_folders]) + 1)
    else:
        opt_num = '1'
    opt_folder = instance_folder + f'opt{opt_num}/'
    try:
        os.mkdir(opt_folder)
    except FileExistsError:
        pass

    # %% 6. Run algorithm
    best_alpha = None
    best_beta = None
    mi = None
    for k in range(5):
        routes_alpha, opt_data_alpha, toolbox_alpha = optimal_route_assignation(fleet, hp_alpha, opt_folder,
                                                                                best_ind=best_alpha,
                                                                                savefig=True,
                                                                                mi=mi,
                                                                                plot_best_generation=False)
        routes_beta, opt_data_beta, toolbox_beta = improve_route(fleet, hp_beta, opt_folder,
                                                                 best_ind=best_beta,
                                                                 savefig=True)
        mi = opt_data_alpha.additional_info['mi']

        if not opt_data_alpha.acceptable: # and not opt_data_beta.acceptable:
            # Not feasible
            print('INCREASING FLEET SIZE BY 1...')
            mi += 1
            '''
            fleet.resize_fleet(len(fleet) + 1)
            best_alpha = opt_data_alpha.bestOfAll

            pos = np.random.randint(len(fleet.network.customers) + len(fleet))
            best_alpha.insert(pos, '|')
            best_alpha.append(np.random.uniform(7 * 60, 10 * 60))
            for i in range(hp_alpha.r):
                chg_op = [-1, sample(fleet.network.charging_stations, 1)[0], uniform(10, 20)]
                index = -len(fleet)
                bestOfAll1 = best_alpha[:index] + chg_op + best_alpha[index:]

            '''
            hp_alpha.num_individuals += 15
            hp_beta.num_individuals += 10
            hp_alpha.max_generations += 10
            hp_beta.max_generations += 10
        else:
            # At least one is acceptable
            break
