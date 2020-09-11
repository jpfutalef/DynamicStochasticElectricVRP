import models.Simulator as Sim
from res.GATools import *
import res.GA_Online as ga

if __name__ == '__main__':
    net_path = '../data/online/21nodes/network.xml'
    fleet_path = '../data/online/21nodes/fleet_assigned.xml'
    collection_path = '../data/online/21nodes/collection.xml'
    mat_path = '../data/online/21nodes/21_nodes.mat'
    report_folder = '../data/online/21nodes/online_no_opt/'

    optimize = True

    ga_time = 1.
    offset_time = 2.
    sample_time = 5.
    sim = Sim.Simulator(net_path, fleet_path, mat_path, collection_path, sample_time, report_folder)

    while not sim.observer.done():
        # Observe
        n, f, current_routes = sim.observer.observe()

        # Optimize if stated
        if optimize:
            CXPB, MUTPB = 0.7, 0.9
            num_individuals = int(len(f.network)) + int(len(f) * 8) + 30
            max_generations = int(num_individuals * 1.2)
            penalization_constant = 10. * len(f.network)
            weights = (0.25, 1.0, 0.5, 1.4, 1.2)  # cost_tt, cost_ec, cost_chg_op, cost_chg_cost, cost_wait_time
            keep_best = 1  # Keep the 'keep_best' best individuals
            tournament_size = 3
            r = 3
            hp = HyperParameters(num_individuals, max_generations, CXPB, MUTPB,
                                 tournament_size=tournament_size,
                                 penalization_constant=penalization_constant,
                                 keep_best=keep_best,
                                 weights=weights,
                                 r=r)
            try:
                os.mkdir(f'../data/online/21nodes/online/')
            except FileExistsError:
                pass
            save_to = f'../data/online/21nodes/online/'

            prev_best, sp = ga.code(f, hp.r)
            routes, f, bestInd, feas, tbb, data = ga.optimal_route_assignation(f, hp, sp, save_to, best_ind=prev_best,
                                                                               savefig=True)
            for ev in f.vehicles.values():
                if len(ev.route[0]) != len(current_routes[ev.id][0]):
                    k = current_routes[ev.id][0].index(ev.route[0][0])
                    S = current_routes[ev.id][0][:k] + ev.route[0]
                    L = current_routes[ev.id][1][:k] + ev.route[1]
                    ev.route = (S, L)
                    ev.assigned_customers = tuple(i for i in S if f.network.isCustomer(i))

            f.write_xml(sim.fleet_path_temp, False, True, True, False)
            sim.fleet = f

        # Move vehicles and modify network
        sim.disturb_network()
        sim.forward_fleet()
