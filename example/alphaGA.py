import os

from models import from_xml
from optimizer import optimal_route_assignation, HyperParameters

# %% 1. Specify instance location and capacities to iterate
folder = 'data/example/'
instance = 'c10cs2_10x10km'
full_path = folder + instance + '.xml'

cs_capacity = 3
soc_policy = (20, 95)

# %% 2. Open instance
print(f'Opening:\n {full_path}')
fleet = from_xml(full_path, assign_customers=False, with_routes=False)

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
                                   algorithm_name='alphaGA')

# %% 5. Specify data folder
# Main instance folder
instance_folder = folder + instance + '/'
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
bestOfAll = None
mi = None
routes, fleet, bestOfAll, feasible, acceptable, toolbox, optData = optimal_route_assignation(fleet,
                                                                                             hyper_parameters,
                                                                                             opt_folder,
                                                                                             best_ind=bestOfAll,
                                                                                             savefig=True,
                                                                                             mi=mi,
                                                                                             plot_best_generation=False)
