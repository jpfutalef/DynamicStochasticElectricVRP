from res.optimizer.GATools import HyperParameters
import res.dispatcher.Dispatcher as Dispatcher
from res.simulator.Simulator import Simulator

"""
GLOBAL PARAMETERS
"""
simulation_name = 'withoutTimeWindows'
start_from = 0
end_at = 50
std_factor = (12., 12.)
soc_policy = (20, 95)
keep = 3

net_path = '../data/online/instance21/init_files/network.xml'
fleet_path = '../data/online/instance21/init_files/fleet.xml'
routes_path = '../data/online/instance21/init_files/routes.xml'
mat_path = '../data/online/instance21/init_files/21_nodes.mat'

onGA_hyper_parameters = HyperParameters(num_individuals=80, max_generations=160, CXPB=0.9, MUTPB=0.6,
                                        weights=(0.1 / 2.218, 1. / 0.4364, 1. / 100, 1. / 500, 1.),
                                        K1=100000, K2=200000, keep_best=1, tournament_size=3, r=2,
                                        alpha_up=soc_policy[1], algorithm_name='onGA', crossover_repeat=1,
                                        mutation_repeat=1)

if __name__ == '__main__':
    """
    WITHOUT OPTIMIZATION
    """

    stage = 'offline'

    for i in range(end_at):
        print(f'--- Simulation ({stage}) #{i} ---')

        main_folder = f'../data/online/instance21/{stage}_{simulation_name}/simulation_{i}/'
        measurements_path = f'../data/online/instance21/{stage}_{simulation_name}/simulation_{i}/measurements.xml'
        history_path = f'../data/online/instance21/{stage}_{simulation_name}/simulation_{i}/history.xml'

        sim = Simulator(net_path, fleet_path, measurements_path, routes_path, history_path, mat_path, 5., main_folder,
                        std_factor=std_factor)

        # Drop time windows
        sim.network.dropTimeWindows()
        sim.network.dropTimeWindows(filepath=sim.network_path)

        non_altered = 0
        while not sim.done():
            if non_altered < keep:
                non_altered += 1
            else:
                sim.disturb_network()
                non_altered = 0
            sim.forward_fleet()
            sim.save_history()

    """ 
    WITH OPTIMIZATION
    """
    stage = 'online'

    for i in range(start_from, end_at):
        print(f'--- Simulation ({stage}) #{i} ---')
        main_folder = f'../data/online/instance21/{stage}_{simulation_name}/simulation_{i}/'
        measurements_path = f'../data/online/instance21/{stage}_{simulation_name}/simulation_{i}/measurements.xml'
        history_path = f'../data/online/instance21/{stage}_{simulation_name}/simulation_{i}/history.xml'
        exec_time_path = f'../../data/online/instance21/{stage}_{simulation_name}/simulation_{i}/exec_time.csv'

        sim = Simulator(net_path, fleet_path, measurements_path, routes_path, history_path, mat_path, 5., main_folder,
                        std_factor=std_factor)

        # Drop time windows
        sim.network.dropTimeWindows(filepath=sim.network_path)

        # Dispatcher to optimize
        dispatcher = Dispatcher.Dispatcher(sim.network_path, sim.fleet_path, sim.measurements_path, sim.routes_path,
                                           onGA_hyper_parameters=onGA_hyper_parameters)

        non_altered = 0
        while not sim.done():
            if non_altered < keep:
                non_altered += 1
            else:
                sim.disturb_network()
                non_altered = 0
            sim.forward_fleet()
            sim.save_history()

            # Optimize
            dispatcher.update()
            dispatcher.optimize_online(exec_time_filepath=exec_time_path)
