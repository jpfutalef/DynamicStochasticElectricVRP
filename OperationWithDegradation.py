from res.GA_Assignation import *
from res.GATools import *
import pandas as pd
import os

# %% 1. Specify instance location, routing frequency, and degradation threshold
data_folder = 'data/instances_real_data/normal_weight/'
instance_filename = '21nodes_0_100_1EV'
# instance_filename = '21nodes_20_95_1EV'
# instance_filename = '21nodes_25_75_1EV'
# instance_filename = '21nodes_25_100_1EV'
# instance_filename = '21nodes_30_70_1EV'
# instance_filename = '21nodes_50_100_1EV'
path = f'{data_folder}{instance_filename}.xml'
print(f'Opening:\n {path}')

route_every = 20  # days
degradation_at = 0.99  # between 0 and 1

# %% 2. Instance fleet
fleet = from_xml(path, assign_customers=False)
fleet.network.draw(save_to=None, width=0.02,
                   edge_color='grey', markeredgecolor='black',
                   markeredgewidth=2.0)

# %% 3. GA hyper-parameters for first routing
CXPB = 0.75
MUTPB = 0.88
num_individuals = 120
max_generations = 1
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

# %% 4. Initialize variables
all_degraded = False
route = False
day = 0

# These store data
capacity_pu: Dict[int, List[float]] = {ev_id: [1.] for ev_id, ev in fleet.vehicles.items()}
capacity_pu_end_day: Dict[int, List[float]] = {ev_id: [1.] for ev_id, ev in fleet.vehicles.items()}
used_capacity_end_day: Dict[int, float] = {ev_id: 0. for ev_id, ev in fleet.vehicles.items()}

# The folder where all results will be saved
opt_folder = f'{data_folder}{instance_filename}/'
try:
    os.mkdir(opt_folder)
except FileExistsError:
    pass

# %% 5. First routes
bestOfAll = None
# bestOfAll = [16, 17, 15, 6, 7, 3, 13, 12, 10, 9, 14, 20, 19, 18, 11, 8, 5, 4, 1, 2, '|', -1, 21, 17.477861132679163, -1, 21, 9.043715167470314, -1, 21, 27.099585092313035, 694.5059771561039]
# bestOfAll = [15, 14, 17, 18, 19, 20, 16, 10, 9, 7, 6, 2, 1, 3, 4, 5, 8, 13, 12, 11, '|', -1, 21, 5.664558477457513, -1, 21, 10.561698891104808, -1, 21, 43.06264480070523, 700.8821086801553]  # 0-100 cambiar 13 por -1
# bestOfAll = [14, 15, 9, 17, 18, 19, 20, 10, 7, 4, 3, 16, 13, 12, 6, 2, 1, 5, 8, 11, '|', 3, 21, 13.151243351255353, -1, 21, 0.2574423237339296, 3, 21, 25.420445559401614, 699.9828181578997] # 25-95
# bestOfAll = [9, 15, 14, 17, 18, 19, 16, 20, 10, 7, 6, 5, 3, 2, 4, 13, 12, 11, 8, 1, '|', -1, 21, 19.700170398095803, 4, 21, 33.87054347634128, 10, 21, 34.92683392101684, 683.9210182855405] # 25-75
# bestOfAll = [9, 15, 14, 18, 19, 17, 6, 4, 3, 2, 10, 5, 16, 20, 8, 7, 1, 12, 13, 11, '|', 17, 21, 28.57100983787813, 20, 21, 25.8966077594467, 10, 21, 38.88255709711892, 636.4233151111349] # 30-70
save_to = f'{opt_folder}day{day}/'
routes, fleet, bestOfAll, feasible, toolbox, optData = optimal_route_assignation(fleet, hyper_parameters, save_to,
                                                                                 best_ind=bestOfAll)

# %% 6. Procedure
while not all_degraded:
    # Route when required
    if (day > 0 and day % route_every == 0) or route:
    #if False:
        save_to = f'{opt_folder}day{day}/'
        if optData.feasible:
            hyper_parameters = HyperParameters(120, 300, CXPB, MUTPB,
                                               tournament_size=tournament_size,
                                               penalization_constant=penalization_constant,
                                               keep_best=keep_best,
                                               weights=weights,
                                               r=r,
                                               starting_points=starting_points)
        else:
            hyper_parameters = HyperParameters(300, 400, CXPB, MUTPB,
                                               tournament_size=tournament_size,
                                               penalization_constant=penalization_constant,
                                               keep_best=keep_best,
                                               weights=weights,
                                               r=r,
                                               starting_points=starting_points)
        routes, fleet, bestOfAll, feasible, toolbox, optData = optimal_route_assignation(fleet, hyper_parameters,
                                                                                         save_to, best_ind=bestOfAll)
        route = False

    # Set routes
    fleet.set_routes_of_vehicles(routes, iterate=False)

    # Degrade
    drop = []
    for ev_id, ev in fleet.vehicles.items():
        eta0 = capacity_pu_end_day[ev_id][-1]

        used_capacity = used_capacity_end_day[ev_id]
        eta, used_capacity = ev.step_degradation_eta_capacity(fleet.network, eta0, used_capacity, fleet.eta_table,
                                                              fleet.eta_model)
        used_capacity_end_day[ev_id] = used_capacity

        #eta = ev.step_degradation_eta(fleet.network, eta0, fleet.eta_table, fleet.eta_model)

        capacity_pu[ev_id] += eta
        capacity_pu_end_day[ev_id] = capacity_pu_end_day[ev_id] + [eta[-1]] if eta else capacity_pu_end_day[ev_id]

        print(f'EV{ev_id} day: {day}  end day capacity:{capacity_pu_end_day[ev_id][-1]}')
        if capacity_pu_end_day[ev_id][-1] <= degradation_at:
            drop.append(ev_id)
            route = True

    for i in drop:
        fleet.drop_vehicle(i)

    if not fleet.vehicles:
        all_degraded = True

    day += 1

# %% Save data and show how long the battery lasted
df_capacity = pd.DataFrame(capacity_pu)
df_capacity.to_csv(f'{opt_folder}capacity_pu.csv')

df_capacity_end_day = pd.DataFrame(capacity_pu_end_day)
df_capacity_end_day.to_csv(f'{opt_folder}capacity_pu_end_day.csv')
