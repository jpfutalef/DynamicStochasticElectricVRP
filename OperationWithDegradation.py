from GA_Assignation import *
from GATools import *
from models.BatteryDegradation import *

# %% 1. Specify instance location
data_folder = 'data/real_data/'
instance_filename = 'data_21nodes_instance_1EV'
path = f'{data_folder}{instance_filename}.xml'

print(f'Opening:\n {path}')

# %% 2. Instance fleet
fleet = from_xml(path, assign_customers=False)
fleet.network.draw(save_to=None, width=0.02,
                   edge_color='grey', markeredgecolor='black',
                   markeredgewidth=2.0)
#input('Ready! Press ENTER to continue...')

# %% 3. GA hyper-parameters
CXPB = 0.55
MUTPB = 0.65
num_individuals = 90
max_generations = 150
penalization_constant = 500000
weights = (0.2, 0.8, 1.2, 0.0)  # travel_time, charging_time, energy_consumption, charging_cost
keep_best = 1  # Keep the 'keep_best' best individuals
tournament_size = 5
r = 4
starting_points = {ev_id: InitialCondition(0, 0, 0, ev.alpha_up, 0) for ev_id, ev in fleet.vehicles.items()}

hyper_parameters = HyperParameters(num_individuals, max_generations, CXPB, MUTPB,
                                   tournament_size=tournament_size,
                                   penalization_constant=penalization_constant,
                                   keep_best=keep_best,
                                   weights=weights,
                                   r=r,
                                   starting_points=starting_points)
print(hyper_parameters)

#input('Press ENTER to continue...')

# %% 4. Run procedure

routes, fleet, bestOfAll, toolbox, optData = optimal_route_assignation(fleet, hyper_parameters, data_folder)



#input('Press ENTER to continue...')