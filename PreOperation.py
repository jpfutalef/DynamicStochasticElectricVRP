import os

from res.models import Fleet, Network
from optimizer.GATools import HyperParameters
from optimizer import alphaGA, betaGA

if __name__ == '__main__':
    # %% 1. Specify instances location
    folder = 'data/online/instance6/source/'
    instances = ['santiago6.xml']

    # %% 2. CS capacities and SOC policy
    cs_capacity = 3
    soc_policy = (20, 95)

    # %% 3. Solve instances
    for instance in instances:
        fleet = Fleet.from_xml(folder + instance, assign_customers=False, with_routes=False, instance=True,
                               from_online=False)

        fleet.modify_cs_capacities(cs_capacity)
        fleet.new_soc_policy(soc_policy[0], soc_policy[1])

        # %% 4. alphaGA hyper-parameters
        num_individuals = int(len(fleet.network) * 1.5) + int(len(fleet) * 10) + 50
        K1 = 100. * len(fleet.network) + 1000 * len(fleet)
        hp_alpha = HyperParameters(num_individuals=num_individuals,
                                   max_generations=num_individuals,
                                   CXPB=0.65,
                                   MUTPB=0.8,
                                   weights=(0.1 / 2.218, 1. / 0.4364, 1. / 100, 1. / 500, 1. * 0.5),
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

        # %% 6. Specify directory to store optimization data
        # A directory to store several optimization results for the same instance
        instance_folder = folder + instance[:-4] + '/'
        try:
            os.mkdir(instance_folder)
        except FileExistsError:
            pass

        # The optimization directory for this optimization
        opt_folders = [os.path.join(instance_folder, d) for d in os.listdir(instance_folder) if
                       os.path.isdir(os.path.join(instance_folder, d))]
        if opt_folders:
            opt_num = str(max([int(i.split('_')[-1]) for i in opt_folders if os.path.isdir(i)]) + 1)
        else:
            opt_num = '1'
        opt_folder = instance_folder + f'opt_{opt_num}/'
        try:
            os.mkdir(opt_folder)
        except FileExistsError:
            pass

        # %% 6. Run algorithm
        best_alpha = None
        best_beta = None
        mi = None
        for k in range(3):
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
