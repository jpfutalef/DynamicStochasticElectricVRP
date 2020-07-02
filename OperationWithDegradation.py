from res.GA_Assignation import *
from res.GATools import *
from models.BatteryDegradation import *
import pandas as pd
import os

# %% 1. Specify instance location
data_folder = 'data/real_data/instances/'
instance_filename = '21nodes_20_95_1EV'
path = f'{data_folder}{instance_filename}.xml'

print(f'Opening:\n {path}')

# %% 2. Instance fleet
fleet = from_xml(path, assign_customers=False)
fleet.network.draw(save_to=None, width=0.02,
                   edge_color='grey', markeredgecolor='black',
                   markeredgewidth=2.0)
# input('Ready! Press ENTER to continue...')

# %% 3. GA hyper-parameters for first routing
CXPB = 0.75
MUTPB = 0.85
num_individuals = 300
max_generations = 450
penalization_constant = 500000
weights = (0.2, 0.8, 1.2, 0.0)  # travel_time, charging_time, energy_consumption, charging_cost
keep_best = 1  # Keep the 'keep_best' best individuals
tournament_size = 3
r = 3
starting_points = {ev_id: InitialCondition(0, 0, 0, ev.alpha_up, 0) for ev_id, ev in fleet.vehicles.items()}

hyper_parameters = HyperParameters(num_individuals, max_generations, CXPB, MUTPB,
                                   tournament_size=tournament_size,
                                   penalization_constant=penalization_constant,
                                   keep_best=keep_best,
                                   weights=weights,
                                   r=r,
                                   starting_points=starting_points)
print(hyper_parameters)

# input('Press ENTER to continue...')

# %% 4. Initialize
degraded = False
day = 0
route_every = 25

# these will store data
stored_eta: Dict[int, list] = {ev_id: [] for ev_id, ev in fleet.vehicles.items()}
cum_eta: Dict[int, float] = {ev_id: 1. for ev_id, ev in fleet.vehicles.items()}

# The folder where all results will be saved
opt_folder = f'{data_folder}{instance_filename}/'
try:
    os.mkdir(opt_folder)
except FileExistsError:
    pass

# %% 5. First routes
bestOfAll = None
#bestOfAll = [18, 15, 10, 9, 14, 17, 19, 20, 16, 4, 1, 3, 2, 6, 7, 5, 8, 13, 12, 11, '|', -1, 21, 10.03621240411417, -1, 21, 3.3093834433210967, -1, 21, 11.178597501344296, 629.6532388963317] # 0-100
#bestOfAll = [14, 15, 9, 17, 18, 19, 20, 10, 7, 4, 3, 16, 13, 12, 6, 2, 1, 5, 8, 11, '|', 3, 21, 13.151243351255353, -1, 21, 0.2574423237339296, 3, 21, 25.420445559401614, 699.9828181578997] # 25-95
#bestOfAll = [9, 15, 14, 17, 18, 19, 16, 20, 10, 7, 6, 5, 3, 2, 4, 13, 12, 11, 8, 1, '|', -1, 21, 19.700170398095803, 4, 21, 33.87054347634128, 10, 21, 34.92683392101684, 683.9210182855405] # 25-75
#bestOfAll = [9, 15, 14, 18, 19, 17, 6, 4, 3, 2, 10, 5, 16, 20, 8, 7, 1, 12, 13, 11, '|', 17, 21, 28.57100983787813, 20, 21, 25.8966077594467, 10, 21, 38.88255709711892, 636.4233151111349] # 30-70
save_to = f'{opt_folder}day{day}/'
routes, fleet, bestOfAll, feasible, toolbox, optData = optimal_route_assignation(fleet, hyper_parameters, save_to,
                                                                                 best_ind=bestOfAll)

# %% 6. Procedure with new hyper-parameters
hyper_parameters = HyperParameters(150, 350, CXPB, MUTPB,
                                   tournament_size=tournament_size,
                                   penalization_constant=penalization_constant,
                                   keep_best=keep_best,
                                   weights=weights,
                                   r=r,
                                   starting_points=starting_points)
while not degraded:
    # New routes
    if day > 0 and day % route_every == 0:
        save_to = f'{opt_folder}day{day}/'
        routes, fleet, bestOfAll, feasible, toolbox, optData = optimal_route_assignation(fleet, hyper_parameters,
                                                                                         save_to, best_ind=bestOfAll)
        if not optData.feasible:
            fleet.plot_operation()
    # Set routes
    fleet.set_routes_of_vehicles(routes)

    # Check degradation after the operation finishes
    for ev_id, ev in fleet.vehicles.items():
        cum_eta[ev_id] = cum_eta[ev_id] * ev.eta
        stored_eta[ev_id].append(cum_eta[ev_id])
        print(f'day: {day}  cum_eta:{cum_eta[ev_id]}')
        if cum_eta[ev_id] <= .8:
            degraded = True
    fleet.set_eta(cum_eta)
    day += 1

days = list(range(day))

# %%
# input('Press ENTER to continue...')

# save data
df_eta = pd.DataFrame(stored_eta)
df_eta.to_csv(f'{opt_folder}eta.csv')
