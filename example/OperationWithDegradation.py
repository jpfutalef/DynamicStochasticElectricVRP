# %% 1. Specify instance location, routing frequency, degradation threshold, and policies
data_folder = 'data/instances_real_data/test_deg_methods/'
instance_filename = '21nodes_0_100'
path = f'{data_folder}{instance_filename}.xml'
print(f'Opening:\n {path}')

route_every = 20  # days
degradation_at = 0.8  # between 0 and 1
policies = [(0, 100), (5, 80), (20, 95), (5, 60), (20, 75), (45, 100)]

for alpha_down, alpha_upp in policies:
    # %% 2. Instance fleet and set policy
    fleet = from_xml(path, assign_customers=False)
    fleet.network.draw(save_to=None, width=0.02,
                       edge_color='grey', markeredgecolor='black',
                       markeredgewidth=2.0)
    fleet.new_soc_policy(alpha_down, alpha_upp)
    print('POLICY: ', alpha_down, alpha_upp)
    # %% 3. GA hyper-parameters for first routing
    CXPB = 0.65
    MUTPB = 0.85
    num_individuals = 250
    max_generations = 350
    penalization_constant = 500000
    weights = (0.2, 0.8, 1.2, 0.0, 3.5)  # travel_time, charging_time, energy_consumption, charging_cost
    keep_best = 1  # Keep the 'keep_best' best individuals
    tournament_size = 3
    r = 3

    hyper_parameters = HyperParameters(num_individuals, max_generations, CXPB, MUTPB,
                                       tournament_size=tournament_size,
                                       penalization_constant=penalization_constant,
                                       keep_best=keep_best,
                                       weights=weights,
                                       r=r)

    # %% 4. Initialize variables
    all_degraded = False
    route = False
    day = 0

    # These store data
    capacity: Dict[int, List[float]] = {ev_id: [1.] for ev_id, ev in fleet.vehicles.items()}
    capacity_end_day: Dict[int, List[float]] = {ev_id: [1.] for ev_id, ev in fleet.vehicles.items()}
    used_capacity_end_day: Dict[int, float] = {ev_id: 0. for ev_id, ev in fleet.vehicles.items()}

    # The folder where all results will be saved
    dod_folder = f'{data_folder}{int(alpha_upp - alpha_down)}/'
    try:
        os.mkdir(dod_folder)
    except FileExistsError:
        pass

    opt_folder = f'{dod_folder}{alpha_down}_{alpha_upp}/'
    try:
        os.mkdir(opt_folder)
    except FileExistsError:
        pass
    # %% 5. First routes
    save_to = f'{opt_folder}day{day}/'
    routes, fleet, bestOfAll, feasible, toolbox, optData = optimal_route_assignation(fleet, hyper_parameters, save_to,
                                                                                     best_ind=None)

    # %% 6. Procedure
    while not all_degraded:
        # Route when required
        if (day > 0 and day % route_every == 0) or route:
            # if False:
            save_to = f'{opt_folder}day{day}/'
            if optData.feasible:
                hyper_parameters = HyperParameters(100, 250, CXPB, MUTPB,
                                                   tournament_size=tournament_size,
                                                   penalization_constant=penalization_constant,
                                                   keep_best=keep_best,
                                                   weights=weights,
                                                   r=r)
            else:
                hyper_parameters = HyperParameters(200, 350, CXPB, MUTPB,
                                                   tournament_size=tournament_size,
                                                   penalization_constant=penalization_constant,
                                                   keep_best=keep_best,
                                                   weights=weights,
                                                   r=r)
            if route:
                routes, fleet, bestOfAll, feasible, toolbox, optData = optimal_route_assignation(fleet, hyper_parameters,
                                                                                                 save_to)
            else:
                routes, fleet, bestOfAll, feasible, toolbox, optData = optimal_route_assignation(fleet, hyper_parameters,
                                                                                                 save_to,
                                                                                                 best_ind=bestOfAll)
            route = False

        # Set routes
        fleet.set_routes_of_vehicles(routes, iterate=False)

        # Degrade
        drop = []
        for ev_id, ev in fleet.vehicles.items():
            # capacity_changes, used_capacity = ev.step_degradation_eta_capacity(fleet.network, used_capacity_end_day[ev_id],
            #                                                                   fleet.eta_table, fleet.eta_model)
            # used_capacity_end_day[ev_id] = used_capacity
            capacity_changes = ev.step_degradation_eta(fleet.network, fleet.eta_table, fleet.eta_model)

            capacity[ev_id] += capacity_changes
            capacity_end_day[ev_id] = capacity_end_day[ev_id] + [capacity_changes[-1]] if capacity_changes else \
            capacity_end_day[ev_id]

            print(f'EV{ev_id} day: {day}  end day capacity:{capacity_end_day[ev_id][-1]}')

            if ev.state_reaching[1, -1] < 0.:
                # Route if an EV battery reached SOC below 0
                route = True

            if capacity_end_day[ev_id][-1] <= degradation_at * ev.battery_capacity_nominal:
                # Route if an EV battery reached EOL
                drop.append(ev_id)
                route = True

        for i in drop:
            fleet.drop_vehicle(i)

        if not fleet.vehicles:
            all_degraded = True

        day += 1

    # %% Save data and show how long the battery lasted
    print('ALL BATTERIES REACHED EOL')
    df_capacity = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in capacity.items() ]))
    df_capacity.to_csv(f'{opt_folder}capacity_pu.csv')

    df_capacity_end_day = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in capacity_end_day.items() ]))
    df_capacity_end_day.to_csv(f'{opt_folder}capacity_pu_end_day.csv')
