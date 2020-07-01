from GA_Assignation import *
from GATools import *
from models.BatteryDegradation import *
import pandas as pd

# %% 1. Specify instance location
data_folder = 'data/real_data/instances/'
instance_filename = '21nodes_NOPOLICY_1EV'
path = f'{data_folder}{instance_filename}.xml'

print(f'Opening:\n {path}')

# %% 2. Instance fleet
fleet = from_xml(path, assign_customers=False)
fleet.network.draw(save_to=None, width=0.02,
                   edge_color='grey', markeredgecolor='black',
                   markeredgewidth=2.0)
#input('Ready! Press ENTER to continue...')

# %% 3. GA hyper-parameters
CXPB = 0.65
MUTPB = 0.8
num_individuals = 90
max_generations = 20
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

degraded = False
stored_eta: Dict[int, list] = {ev_id: [] for ev_id, ev in fleet.vehicles.items()}
cum_eta: Dict[int, float] = {ev_id: 1. for ev_id, ev in fleet.vehicles.items()}
day = 1

# these will store data
routes_list = []
ind_list = []
opt_folder = f'{data_folder}{instance_filename}/'

while not degraded:
    routes, fleet, bestOfAll, toolbox, optData = optimal_route_assignation(fleet, hyper_parameters, opt_folder)
    routes_list.append(routes)
    ind_list.append(bestOfAll)
    for ev_id, ev in fleet.vehicles.items():
        stored_eta[ev_id].append(ev.eta)
        cum_eta[ev_id] = cum_eta[ev_id]*np.prod(ev.eta)
        print('cum_eta', cum_eta[ev_id])
        if cum_eta[ev_id] <= .9995:
            degraded = True
    fleet.set_eta(cum_eta)
    day += 1

days = list(range(day))


#input('Press ENTER to continue...')