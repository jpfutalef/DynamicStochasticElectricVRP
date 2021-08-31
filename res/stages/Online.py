import datetime, sys
from pathlib import Path
from typing import NamedTuple, Union, Tuple
from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt

import res.simulator.Simulator as Simulator
import res.dispatcher.Dispatcher as Dispatcher
from res.optimizer.GATools import OnGA_HyperParameters as HyperParameters


class OnlineParameters(NamedTuple):
    source_folder: Path
    simulation_folder: Path
    optimize: bool
    hyper_parameters: Union[HyperParameters, None]
    keep_times: int = 1
    sample_time: Union[float, int] = 5.0
    std_factor: float = 1.0


def setup_simulator(source_folder: Path, save_to_folder: Path, sample_time=180.0, std_factor=(1., 1.),
                    fleet_filepath: Path = None,
                    previous_day_measurements: Dict[int, Dispatcher.ElectricVehicleMeasurement] = None,
                    routes_filepath: Path = None, network_filepath=None, mat_filepath=None, new_soc_policy=None,
                    include_mat=False):
    for file in source_folder.iterdir():
        if mat_filepath is None and (file.stem == 'data' or file.suffix == '.mat'):
            mat_filepath = file
        elif file.stem == 'network':
            network_filepath = file
        elif routes_filepath is None and file.stem == 'routes':
            routes_filepath = file
        elif fleet_filepath is None and file.stem == 'fleet':
            fleet_filepath = file

    measurements_filepath = Path(save_to_folder, 'measurements.xml')
    history_filepath = Path(save_to_folder, 'history.xml')
    if include_mat:
        sim = Simulator.Simulator(network_filepath, fleet_filepath, measurements_filepath, routes_filepath,
                                  history_filepath, mat_filepath, sample_time, save_to_folder, std_factor,
                                  create_routes_xml=True, create_measurements_xml=True, create_history_xml=True,
                                  previous_day_measurements=previous_day_measurements, new_soc_policy=new_soc_policy)
    else:
        sim = Simulator.Simulator(network_filepath, fleet_filepath, measurements_filepath, routes_filepath,
                                  history_filepath, None, sample_time, save_to_folder, std_factor,
                                  create_routes_xml=True, create_measurements_xml=True, create_history_xml=True,
                                  previous_day_measurements=previous_day_measurements, new_soc_policy=new_soc_policy)
    return sim


def online_operation_open_loop(p: OnlineParameters):
    simulator = setup_simulator(p.source_folder, p.simulation_folder, p.sample_time, p.std_factor)
    # Start loop
    dont_alter = 0
    while not simulator.done():
        if not dont_alter:
            simulator.disturb_network()
        dont_alter = dont_alter + 1 if dont_alter + 1 < p.keep_times else 0
        simulator.forward_fleet()
        simulator.save_history()
    return 1


def online_operation_closed_loop(p: OnlineParameters):
    simulator = setup_simulator(p.source_folder, p.simulation_folder, p.sample_time, p.std_factor)
    dispatcher = Dispatcher.Dispatcher(simulator.network_path, simulator.fleet_path, simulator.measurements_path,
                                       simulator.routes_path, onGA_hyper_parameters=p.hyper_parameters)
    exec_time_path = Path(p.simulation_folder, 'exec_time.csv')
    # Start loop
    dont_alter = 0
    while not simulator.done():
        if not dont_alter:
            simulator.disturb_network()
            dispatcher.update()
            dispatcher.optimize_online(exec_time_filepath=exec_time_path)
        dont_alter = dont_alter + 1 if dont_alter + 1 < p.keep_times else 0
        simulator.forward_fleet()
        simulator.save_history()
    return


def online_operation(source_folder: Path, simulations_folder: Path, hp: HyperParameters, optimize: bool = False,
                     repetitions: int = 5, keep_times: int = 4, sample_time: float = 2.0,
                     std_factor: float = 1.0):
    for i in range(repetitions):
        now = datetime.datetime.now().strftime("%H-%M-%S_%d-%m-%y")
        simulation_folder = Path(simulations_folder, now)
        p = OnlineParameters(source_folder, simulation_folder, optimize, hp, keep_times, sample_time,
                             std_factor)
        print(f'Initializing simulation #{i + 1} in folder: /{simulation_folder.stem}')
        if p.optimize:
            online_operation_closed_loop(p)
        else:
            online_operation_open_loop(p)
        print('Done!\n')


def online_operation_degradation(source_folder: Path, main_folder: Path, hp: HyperParameters, eta_table: np.ndarray,
                                 keep_times: int = 4, sample_time: float = 2.0,
                                 std_factor: Tuple[float, float] = (1., 1.), policy=(20., 95.), degrade_until=0.8):
    eta_model = NearestNeighbors(n_neighbors=3).fit(eta_table[:, 0:2])
    now = datetime.datetime.now().strftime("%H-%M-%S_%d-%m-%y")

    policy_folder = Path(main_folder, f'policy_{policy[0]}_{policy[1]}')
    simulation_folder = Path(policy_folder, now)

    day = 1
    fleet_filepath = None
    routes_filepath = None
    previous_day_measurements = None

    while True:
        day_operation_folder = Path(simulation_folder, f'day_{day}')
        simulator = setup_simulator(source_folder, day_operation_folder, sample_time, std_factor, fleet_filepath,
                                    previous_day_measurements=previous_day_measurements,
                                    routes_filepath=routes_filepath,
                                    new_soc_policy=policy)
        simulator.eta_model = eta_model
        simulator.eta_table = eta_table
        dispatcher = Dispatcher.Dispatcher(simulator.network_path, simulator.fleet_path,
                                           simulator.measurements_path, simulator.routes_path,
                                           onGA_hyper_parameters=hp)
        exec_time_path = Path(day_operation_folder, 'exec_time.csv')
        # Start loop
        dont_alter = 0
        while not simulator.done():
            if not dont_alter:
                simulator.disturb_network()
                dispatcher.update()
                dispatcher.optimize_online(exec_time_filepath=exec_time_path)
            dont_alter = dont_alter + 1 if dont_alter + 1 < keep_times else 0
            simulator.forward_fleet()
            simulator.save_history()

        # If degraded, stop
        if not simulator.fleet.healthy(degrade_until):
            break

        # Setup to begin a new day
        fleet_filepath = Path(day_operation_folder, 'fleet_end.xml')
        simulator.fleet.write_xml(fleet_filepath, print_pretty=False)

        routes_filepath = Path(day_operation_folder, 'routes_end.xml')
        simulator.write_routes(routes_filepath)
        previous_day_measurements = simulator.measurements
        day += 1
    return
