from res.optimizer.GATools import HyperParameters
import res.dispatcher.Dispatcher as Dispatcher
from res.simulator.Simulator import Simulator
from os import makedirs
from datetime import datetime

"""
Configuration
"""
simulation_name = 'withoutTimeWindows'
num_of_simulations = 10
soc_policy = (20, 95)
keep_num_realizations = 1
online_verbose = False

folder_path = '/data/online/case1/'
net_path = f'{folder_path}network_C10_CS4_ttSTD_1.0_ecSTD1.0'
fleet_path = f'{folder_path}fleet.xml'
routes_path = f'{folder_path}routes.xml'

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

    for i in range(num_of_simulations):
        print(f'--- Simulation ({stage}) #{i} ---')

        main_folder = f'data/online/instance21/{stage}_{simulation_name}/simulation_{i}/'
        measurements_path = f'data/online/instance21/{stage}_{simulation_name}/simulation_{i}/measurements.xml'
        history_path = f'data/online/instance21/{stage}_{simulation_name}/simulation_{i}/history.xml'

        sim = Simulator(net_path, fleet_path, measurements_path, routes_path, history_path, mat_path, 5., main_folder)

        # Start loop
        non_altered = 0
        while not sim.done():
            if non_altered < keep_num_realizations:
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

    for i in range(48, 50):
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

        # Drop time windows and save
        sim.network.drop_time_windows(filepath=sim.network_path)

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