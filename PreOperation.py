import os

import res.optimizer.GATools as GATools
from res.models import Fleet
from res.optimizer import alphaGA

# %% Configuration
folder = 'data/test/'
instance_name = 'fleet2'
results_folder = None  # folder + 'results/'
max_add_vehicles = 0
weights = (0.1, 10., 1e-1, 1e-4)
additional_vehicles = 2
fill_up_to = 0.5
fleet_size = None

hp_alpha = GATools.AlphaGA_HyperParameters(weights, CXPB=0.65, MUTPB=0.85, hard_penalization=200000,
                                           elite_individuals=1, tournament_size=5, r=3)
hp_beta = GATools.BetaGA_HyperParameters(weights, CXPB=0.6, MUTPB=0.8, hard_penalization=200000, elite_individuals=1,
                                         tournament_size=5)


if __name__ == '__main__':
    for i in range(3):
        # %% Read instance data
        fleet = Fleet.from_xml(folder + instance_name + '.xml')
        soc_policy = (fleet.vehicles[0].alpha_down, fleet.vehicles[0].alpha_up)

        # %% Initial population and fleet size
        N = int(sum([fleet.network.demand(i) for i in fleet.network.customers])/fleet.vehicles[0].max_payload) + 1
        fill_up_to = N/(N+additional_vehicles) if additional_vehicles else fill_up_to
        init_pop, m = alphaGA.heuristic_population_1(hp_alpha.r, fleet, fill_up_to)
        fleet.resize_fleet(m)
        hp_alpha.num_individuals = 10*len(fleet) + 5*len(fleet.network) + 10
        hp_alpha.max_generations = 3*hp_alpha.num_individuals + 15
        hp_alpha.alpha_up = fleet.vehicles[0].alpha_up

        results_folder = folder + instance_name
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

        # %% Optimize
        routes_alpha, opt_data_alpha, toolbox_alpha = alphaGA.alphaGA(fleet, hp_alpha, opt_folder,
                                                                      init_pop=init_pop,
                                                                      savefig=True,
                                                                      plot_best_generation=False)
