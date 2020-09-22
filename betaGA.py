# %% Imports
import os

from models.OnlineFleet import from_xml
from res.betaGA import optimal_route_assignation, HyperParameters

# %% 1. Specify instance location and capacities to iterate
folder = 'data/test/c10cs2_10x10km/opt1/alphaGA_fleetsize_1/'
instance = 'result_instance'
full_path = folder + instance + '.xml'

cs_capacity = 3
soc_policy = (20, 95)

# %% 2. Open instance
print(f'Opening:\n {full_path}')
fleet = from_xml(full_path, assign_customers=True, with_routes=True)

# %% 3. Modify CS capacities and
fleet.modify_cs_capacities(cs_capacity)

# %% 4. Set soc policy
fleet.new_soc_policy(soc_policy[0], soc_policy[1])

# %% 4. GA hyper-parameters
num_individuals = int(len(fleet.network) * 1.5) + int(len(fleet) * 10) + 50
K1 = 100. * len(fleet.network) + 1000 * len(fleet)
hyper_parameters = HyperParameters(num_individuals=num_individuals,
                                   max_generations=num_individuals * 3,
                                   CXPB=0.7,
                                   MUTPB=0.9,
                                   weights=(0.5/2.218, 1./0.4364, 1./8, 1./80, 1.2),
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
folder_list = folder.split('/')
opt_folder = f'{folder_list[0]}/{folder_list[1]}/{folder_list[2]}/{folder_list[3]}/'

# %% 6. Run algorithm
bestOfAll = [4, 8, 9, 3, 7, 10, 1, 5, 6, 2, -1, 15.588534737915905, -1, 14.101247887292242, -1, 11.78015138126318, -1,
             6.353721881939879, -1, 11.39928029790051, -1, 5.056704874730546, -1, 9.025212354969659, -1,
             17.88367184069846, -1, 12.702030035068944, -1, 9.64825997203456, 0.0]
routes, opt_data, toolbox = optimal_route_assignation(fleet, hyper_parameters, opt_folder, best_ind=bestOfAll,
                                                      savefig=True)
