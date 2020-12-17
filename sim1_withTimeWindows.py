from res.optimizer.GATools import HyperParameters
import res.dispatcher.Dispatcher as Dispatcher
from res.simulator.Simulator import Simulator
from os import makedirs
from datetime import datetime

"""
GLOBAL PARAMETERS
"""
simulation_name = 'withTimeWindows'
start_from = 0
end_at = 50
std_factor = (3., 3.)
soc_policy = (20, 95)
keep = 1
log_online = False

net_path = 'data/online/instance21/init_files/network.xml'
fleet_path = 'data/online/instance21/init_files/fleet.xml'
routes_path = 'data/online/instance21/init_files/routes.xml'
mat_path = 'data/online/instance21/init_files/21_nodes.mat'

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

    for i in range(start_from, end_at):
        print(f'--- Simulation ({stage}) #{i} ---')

        main_folder = f'data/online/instance21/{stage}_{simulation_name}/simulation_{i}/'
        measurements_path = f'data/online/instance21/{stage}_{simulation_name}/simulation_{i}/measurements.xml'
        history_path = f'data/online/instance21/{stage}_{simulation_name}/simulation_{i}/history.xml'

        sim = Simulator(net_path, fleet_path, measurements_path, routes_path, history_path, mat_path, 5., main_folder,
                        std_factor=std_factor)

        # Start loop
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
        main_folder = f'data/online/instance21/{stage}_{simulation_name}/simulation_{i}/'
        measurements_path = f'{main_folder}/measurements.xml'
        history_path = f'{main_folder}/history.xml'
        exec_time_path = f'{main_folder}/exec_time.csv'

        # Log folders
        log_measurements_folder = main_folder + 'logs/measurements/'
        log_routes_folder = main_folder + 'logs/routes/'
        log_histories_folder = main_folder + 'logs/histories/'

        makedirs(log_measurements_folder, exist_ok=True)
        makedirs(log_routes_folder, exist_ok=True)
        makedirs(log_histories_folder, exist_ok=True)

        # Simulation object
        sim = Simulator(net_path, fleet_path, measurements_path, routes_path, history_path, mat_path, 5.,
                        main_folder=main_folder, std_factor=std_factor)

        # Dispatcher to optimize
        dispatcher = Dispatcher.Dispatcher(sim.network_path, sim.fleet_path, sim.measurements_path, sim.routes_path,
                                           onGA_hyper_parameters=onGA_hyper_parameters)

        # Start loop
        non_altered = 0
        while not sim.done():
            if non_altered < keep:
                non_altered += 1
            else:
                sim.disturb_network()
                non_altered = 0

            sim.forward_fleet()
            sim.save_history()

            # Time when everything occurs
            log_time = datetime.today().strftime('%Y_%m_%d-%H_%M_%S')

            # Log measurements
            if log_online:
                log_measurements_path = log_measurements_folder + log_time + '.xml'
                Dispatcher.write_measurements(log_measurements_path, sim.measurements)

            # Optimize
            dispatcher.update()
            dispatcher.optimize_online(exec_time_filepath=exec_time_path)

            # Log routes after optimization
            if log_online:
                log_routes_path = log_routes_folder + log_time + '.xml'
                Dispatcher.write_routes(log_routes_path, dispatcher.routes, dispatcher.depart_info)

