import pandas as pd
import datetime

from res.optimizer.GATools import HyperParameters
from res.tools.IOTools import create_folder
from res.simulator import Simulator
import res.optimizer.onGA as ga

# %% 1. Specify instances location
main_folder = 'data/online/instance21/'
mat_path = 'data/online/instance21/21_nodes.mat'
net_path = 'data/online/instance21/network.xml'
fleet_path = 'data/online/instance21/fleet.xml'
num_iterations = 50

ga_time = 1.
offset_time = 2.
sample_time = 5.
soc_policy = (20, 95)

for i in range(num_iterations):
    iteration_folder = f'{main_folder}iteration_{i + 1}/'
    no_opt_folder = f'{iteration_folder}no_opt/'
    opt_folder = f'{iteration_folder}opt/'
    collection_no_opt_path = f'{no_opt_folder}collection.xml'
    collection_opt_path = f'{opt_folder}collection.xml'

    create_folder(iteration_folder)
    create_folder(no_opt_folder)
    create_folder(opt_folder)

    # OPERATE WITHOUT OPTIMIZATION

    sim = Simulator(net_path, fleet_path, collection_no_opt_path, mat_path, sample_time)
    while not sim.dispatcher.done():
        # Disturb network
        sim.disturb_network()

        # Forward vehicles
        sim.forward_fleet()

    nopt_df = pd.DataFrame({i: [sum(j)] for i, j in sim.violations.items()})

    # OPERATE WITH OPTIMIZATION
    sim = Simulator(net_path, fleet_path, collection_opt_path, mat_path, sample_time)
    while not sim.dispatcher.done():
        # Disturb network
        sim.disturb_network()

        # Observe
        n, f, f_original, current_routes, ahead_routes = sim.dispatcher.synchronization()

        if not ahead_routes:
            break

        # Optimize
        num_individuals = int(len(f.network) * 1.5) + int(len(f) * 10) + 50
        K1 = 100. * len(f.network) + 1000 * len(f)
        hp = HyperParameters(num_individuals=num_individuals,
                             max_generations=num_individuals * 3,
                             CXPB=0.7,
                             MUTPB=0.9,
                             weights=(0.5 / 2.218, 1. / 0.4364, 1. / 8, 1. / 80, 1.2),
                             K1=K1,
                             K2=K1 * 2.5,
                             keep_best=1,
                             tournament_size=3,
                             r=4,
                             alpha_up=soc_policy[1],
                             algorithm_name='onGA',
                             crossover_repeat=1,
                             mutation_repeat=1)

        prev_best, critical_points = ga.code(f, ahead_routes, hp.r)
        now = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        save_opt_to = f'{opt_folder}{now}/'
        routes, opt_data, toolbox = ga.onGA(f, hp, critical_points, save_opt_to,
                                            best_ind=prev_best, savefig=True)

        # Modify current route
        new_routes = {}
        for id_ev, ((S_old, L_old), x10, x20, x30) in current_routes.items():
            if id_ev in routes.keys():
                S_new, L_new = routes[id_ev][0]
                k = S_old.index(S_new[0])
                new_routes[id_ev] = ((S_old[:k] + S_new, L_old[:k] + L_new), x10, x20, x30)

        # Send routes to vehicles
        f.set_routes_of_vehicles(new_routes, iterate=False)
        f_original.update_from_another_fleet(f)
        f_original.write_xml(sim.dispatcher.fleet_path, online=True, assign_customers=True)

        # Forward vehicles
        sim.forward_fleet()

    opt_df = pd.DataFrame({i: [sum(j)] for i, j in sim.violations.items()})

    # SAVE RESULTS
    with pd.ExcelWriter(f'{iteration_folder}violations.xlsx') as writer:
        nopt_df.to_excel(writer, sheet_name='NoOpt')
        opt_df.to_excel(writer, sheet_name='Opt')
