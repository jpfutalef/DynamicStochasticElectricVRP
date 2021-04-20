import os

from res.models import Fleet, Network
from res.optimizer.GATools import HyperParameters
from res.optimizer import alphaGA, betaGA

if __name__ == '__main__':
    # %% 1. Specify instances location
    folder = 'data/instances/'
    results_folder = None  # folder + 'results/'
    instance_name = 'network_C10_CS4_ttSTD_1.0_ecSTD1.0'
    fleet_name = 'fleet'

    # %% 2. Configuration
    deterministic = False
    cs_capacity = 3
    soc_policy = (20, 95)
    fleet_size = 1

    for i in range(3):
        max_add_vehicles = 0

        # %% 3. Read data
        fleet = Fleet.from_xml(folder + fleet_name + '.xml')
        network = Network.from_xml(folder + instance_name + '.xml')

        fleet.set_network(network)
        fleet.modify_cs_capacities(cs_capacity)
        fleet.new_soc_policy(soc_policy[0], soc_policy[1])
        fleet.resize_fleet(fleet_size)
        fleet.deterministic = deterministic

        # %% 4. alphaGA hyper-parameters
        num_individuals = int(len(fleet.network) * 1.5) + int(len(fleet) * 10) + 120
        K1 = 100. * len(fleet.network) + 1000 * len(fleet)
        hp_alpha = HyperParameters(num_individuals=num_individuals,
                                   max_generations=6*num_individuals,
                                   CXPB=0.65,
                                   MUTPB=0.85,
                                   weights=(0.1, 10., 0.001, 0.001, 1.),
                                   K1=K1,
                                   K2=K1 * 2.5,
                                   keep_best=1,
                                   tournament_size=3,
                                   r=4,
                                   alpha_up=soc_policy[1],
                                   algorithm_name='alphaGA')

        # %% 5. betaGA hyper-parameters
        num_individuals = int(len(fleet.network) * 1.5) + int(len(fleet) * 10) + 50
        K1 = 100. * len(fleet.network) + 1000 * len(fleet)
        hp_beta = HyperParameters(num_individuals=num_individuals,
                                  max_generations=num_individuals * 3,
                                  CXPB=0.7,
                                  MUTPB=0.9,
                                  weights=(0.1 / 2.218, 1. / 0.4364, 1. / 100, 1. / 500, 1. * 0.5),
                                  K1=K1,
                                  K2=K1 * 2.5,
                                  keep_best=1,
                                  tournament_size=3,
                                  r=2,
                                  alpha_up=soc_policy[1],
                                  algorithm_name='betaGA',
                                  crossover_repeat=2,
                                  mutation_repeat=2)

        # %% 6. Specify directories
        # If not results folder, create one for the instance
        if results_folder is None and fleet.deterministic:
            results_folder = folder + instance_name + '_deterministic/'

        elif results_folder is None and not fleet.deterministic:
            results_folder = folder + instance_name + '_stochastic/'


        # A directory to store several optimization results for the same instance
        try:
            os.mkdir(results_folder)
        except FileExistsError:
            pass

        # The optimization directory for this optimization
        opt_folders = [os.path.join(results_folder, d) for d in os.listdir(results_folder) if
                       os.path.isdir(os.path.join(results_folder, d))]
        if opt_folders:
            opt_num = str(max([int(i.split('_')[-1]) for i in opt_folders if os.path.isdir(i)]) + 1)
        else:
            opt_num = '1'
        opt_folder = results_folder + f'opt_{opt_num}/'
        try:
            os.mkdir(opt_folder)
        except FileExistsError:
            pass

        # %% 6. Deterministic optimization
        best_alpha = None
        best_beta = None
        mi = None
        for k in range(max_add_vehicles + 1):
            routes_alpha, opt_data_alpha, toolbox_alpha = alphaGA.alphaGA(fleet, hp_alpha, opt_folder,
                                                                          best_ind=best_alpha,
                                                                          savefig=True,
                                                                          mi=mi,
                                                                          plot_best_generation=False)

            routes_beta, opt_data_beta, toolbox_beta = betaGA.betaGA(fleet, hp_beta, opt_folder,
                                                                     best_ind=best_beta,
                                                                     savefig=True)

            mi = opt_data_alpha.additional_info['mi']

            if not opt_data_alpha.acceptable and not opt_data_beta.acceptable:
                # Both alphaGA and betaGA are not feasible
                print('INCREASING FLEET SIZE BY 1...')
                mi += 1
                hp_alpha.num_individuals += 15
                hp_beta.num_individuals += 10
                hp_alpha.max_generations += 10
                hp_beta.max_generations += 10
            else:
                # At least one is acceptable
                break

    # %% Last one
    max_add_vehicles = 3

    # %% 3. Read data
    fleet = Fleet.from_xml(folder + fleet_name + '.xml')
    network = Network.from_xml(folder + instance_name + '.xml')

    fleet.set_network(network)
    fleet.modify_cs_capacities(cs_capacity)
    fleet.new_soc_policy(soc_policy[0], soc_policy[1])
    fleet.resize_fleet(fleet_size)
    fleet.deterministic = deterministic

    # %% 4. alphaGA hyper-parameters
    num_individuals = int(len(fleet.network) * 1.5) + int(len(fleet) * 10) + 90
    K1 = 100. * len(fleet.network) + 1000 * len(fleet)
    hp_alpha = HyperParameters(num_individuals=num_individuals,
                               max_generations=6 * num_individuals,
                               CXPB=0.65,
                               MUTPB=0.85,
                               weights=(0.1, 10., 0.001, 0.001, 1.),
                               K1=K1,
                               K2=K1 * 2.5,
                               keep_best=1,
                               tournament_size=3,
                               r=4,
                               alpha_up=soc_policy[1],
                               algorithm_name='alphaGA')

    # %% 5. betaGA hyper-parameters
    num_individuals = int(len(fleet.network) * 1.5) + int(len(fleet) * 10) + 50
    K1 = 100. * len(fleet.network) + 1000 * len(fleet)
    hp_beta = HyperParameters(num_individuals=num_individuals,
                              max_generations=num_individuals * 3,
                              CXPB=0.7,
                              MUTPB=0.9,
                              weights=(0.1 / 2.218, 1. / 0.4364, 1. / 100, 1. / 500, 1. * 0.5),
                              K1=K1,
                              K2=K1 * 2.5,
                              keep_best=1,
                              tournament_size=3,
                              r=2,
                              alpha_up=soc_policy[1],
                              algorithm_name='betaGA',
                              crossover_repeat=2,
                              mutation_repeat=2)

    # %% 6. Specify directories
    # If not results folder, create one for the instance
    if results_folder is None and fleet.deterministic:
        results_folder = folder + instance_name + '_deterministic/'

    elif results_folder is None and not fleet.deterministic:
        results_folder = folder + instance_name + '_stochastic/'

    # A directory to store several optimization results for the same instance
    try:
        os.mkdir(results_folder)
    except FileExistsError:
        pass

    # The optimization directory for this optimization
    opt_folders = [os.path.join(results_folder, d) for d in os.listdir(results_folder) if
                   os.path.isdir(os.path.join(results_folder, d))]
    if opt_folders:
        opt_num = str(max([int(i.split('_')[-1]) for i in opt_folders if os.path.isdir(i)]) + 1)
    else:
        opt_num = '1'
    opt_folder = results_folder + f'opt_{opt_num}/'
    try:
        os.mkdir(opt_folder)
    except FileExistsError:
        pass

    # %% 6. Deterministic optimization
    best_alpha = None
    best_beta = None
    mi = None
    for k in range(max_add_vehicles + 1):
        routes_alpha, opt_data_alpha, toolbox_alpha = alphaGA.alphaGA(fleet, hp_alpha, opt_folder,
                                                                      best_ind=best_alpha,
                                                                      savefig=True,
                                                                      mi=mi,
                                                                      plot_best_generation=False)

        routes_beta, opt_data_beta, toolbox_beta = betaGA.betaGA(fleet, hp_beta, opt_folder,
                                                                 best_ind=best_beta,
                                                                 savefig=True)

        mi = opt_data_alpha.additional_info['mi']

        if not opt_data_alpha.acceptable and not opt_data_beta.acceptable:
            # Both alphaGA and betaGA are not feasible
            print('INCREASING FLEET SIZE BY 1...')
            mi += 1
            hp_alpha.num_individuals += 15
            hp_beta.num_individuals += 10
            hp_alpha.max_generations += 10
            hp_beta.max_generations += 10
        else:
            # At least one is acceptable
            break
