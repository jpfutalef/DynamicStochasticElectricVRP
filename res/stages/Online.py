import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Tuple, Type

import numpy as np
from sklearn.neighbors import NearestNeighbors

from res.dispatcher import Dispatcher
from res.optimizer.GATools import OnGA_HyperParameters as HyperParameters
from res.results import OnlineResults
from res.simulator import Simulator
from res.models import Fleet, Network


@dataclass
class OnlineParameters:
    source_folder: Path
    simulation_folder: Path
    optimize: bool
    soc_policy: Tuple[float, float]
    hyper_parameters: Union[HyperParameters, None]
    keep_times: int = 1
    sample_time: Union[float, int] = 300
    std_factor: float = 1.0
    start_earlier_by: float = 600.
    ev_type: Type[Fleet.EV.ElectricVehicle] = None
    fleet_type: Type[Fleet.Fleet] = None
    edge_type: Type[Network.Edge.DynamicEdge] = None
    network_type: Type[Network.Network] = None

    network_path: Path = None
    fleet_path: Path = None
    routes_path: Path = None
    eta_model: NearestNeighbors = None
    eta_table: np.ndarray = None

    def __post_init__(self):
        self.fleet_path = Path(self.source_folder, "instance.xml")
        self.network_path = Path(self.source_folder, "instance.xml")
        if not self.fleet_path.is_file():
            self.fleet_path = Path(self.source_folder, "fleet.xml")
            self.network_path = Path(self.source_folder, "network.xml")
        self.routes_path = Path(self.source_folder, "routes.xml")

    def __str__(self):
        s = ""
        skip = ["hyper_parameters", "eta_model", "eta_table"]
        for (key, val) in {x: y for x, y in self.__dict__.items() if x not in skip}:
            s += f"        {key}:  {val}\n"
        return s


def online_operation(main_folder: Path, source_folder: Path, optimize: bool = False, onGA_hp: HyperParameters = None,
                     repetitions: int = 5, hold_by: int = 0, sample_time: float = 300.,
                     std_factor: float = 1.0, start_earlier_by: float = 600,
                     soc_policy: Tuple[float, float] = (20., 95), display_gui: bool = False,
                     ev_type: Type[Fleet.EV.ElectricVehicle] = Fleet.EV.ElectricVehicle,
                     fleet_type: Type[Fleet.Fleet] = Fleet.Fleet,
                     edge_type: Type[Network.Edge.DynamicEdge] = Network.Edge.DynamicEdge,
                     network_type: Type[Network.Network] = Network.Network):
    simulation_type = 'closed_loop' if optimize else 'open_loop'
    simulations_folder = Path(main_folder, simulation_type)
    if display_gui:
        if not source_folder.is_dir():
            print("Directory is not valid: ", source_folder)
            return 0
        print("Will simulate results from:\n  ", source_folder)
        print("Simulation results will be saved to:\n  ", simulations_folder)
        input("Press ENTER to continue... (ctrl+Z to end process)")

    simulations_folder = Path()
    for i in range(repetitions):
        simulations_folder = simulate(main_folder, source_folder, optimize, onGA_hp, hold_by, sample_time, std_factor,
                                      start_earlier_by, soc_policy, display_gui, None, None, ev_type, fleet_type,
                                      edge_type, network_type)

    # Summarize results
    df_costs, df_constraints = OnlineResults.folder_data(simulations_folder.parent, source_folder)
    df_costs.to_csv(Path(simulations_folder.parent, 'costs.csv'))
    df_constraints.to_csv(Path(simulations_folder.parent, 'constraints.csv'))


def simulate(main_folder: Path, source_folder: Path = None, optimize: bool = False, onGA_hp: HyperParameters = None,
             hold_by: int = 0, sample_time: float = 300.0, std_factor: float = 1.0, start_earlier_by: float = 600,
             soc_policy: Tuple[float, float] = (20., 95), display_gui: bool = True,
             previous_simulation_folder: Path = None, simulation_name: str = None,
             ev_type: Type[Fleet.EV.ElectricVehicle] = Fleet.EV.ElectricVehicle,
             fleet_type: Type[Fleet.Fleet] = Fleet.Fleet,
             edge_type: Type[Network.Edge.DynamicEdge] = Network.Edge.DynamicEdge,
             network_type: Type[Network.Network] = Network.Network):
    if simulation_name is None:
        simulation_name = datetime.datetime.now().strftime("%H-%M-%S_%d-%m-%y")
    simulation_type = 'closed_loop' if optimize else 'open_loop'
    simulations_folder = Path(main_folder, simulation_type)
    simulation_folder = Path(simulations_folder, simulation_name)

    p = OnlineParameters(source_folder, simulation_folder, optimize, soc_policy, onGA_hp, hold_by, sample_time,
                         std_factor, start_earlier_by, ev_type, fleet_type, edge_type, network_type)
    if display_gui:
        text = f"""The simulation will run using the following parameters:
{p}
Simulation results will be saved to:
        {simulation_folder}
Press any key to continue..."""
        input(text)
    else:
        print(f"Initializing simulation at {simulation_folder}")

    if optimize and onGA_hp is None:
        raise ValueError("Optimization was requested, but no hyper parameters were given.")
    elif optimize:
        online_operation_closed_loop(p, previous_simulation_folder)
    else:
        online_operation_open_loop(p, previous_simulation_folder)
    print('Done!')
    return simulation_folder


def online_operation_open_loop(p: OnlineParameters, previous_simulation_folder: Path = None, history_figsize=(16, 5)):
    # Create simulator instance
    simulator = setup_simulator(p, previous_simulation_folder)

    # Disturb to create first network realization
    simulator.disturb_network()
    simulator.save_network()

    # Start loop
    while not simulator.done():
        simulator.forward_fleet()
        simulator.save_history()

    # Save history figures
    simulator.save_history_figures(history_figsize=history_figsize)
    return


def online_operation_closed_loop(p: OnlineParameters, previous_simulation_folder: Path = None, history_figsize=(16, 5)):
    # Create simulator instance
    simulator = setup_simulator(p, previous_simulation_folder)

    # Create dispatcher instance
    dispatcher = Dispatcher.Dispatcher(simulator.network_path, simulator.fleet_path, simulator.measurements_path,
                                       simulator.routes_path, p.ev_type, p.fleet_type, p.edge_type, p.network_type,
                                       p.hyper_parameters)

    exec_time_path = Path(p.simulation_folder, 'exec_time.csv')

    # Disturb to create first network realization
    simulator.disturb_network()
    simulator.save_network()

    # Start loop
    while not simulator.done():
        # Measure and update
        dispatcher.update()
        dispatcher.optimize_online(exec_time_filepath=exec_time_path)

        # Simulate one step
        simulator.forward_fleet()
        simulator.save_history()

    # Save history figures
    simulator.save_history_figures(history_figsize=history_figsize)
    return


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


def setup_simulator(op: OnlineParameters, previous_simulation_folder: Path = None):
    if previous_simulation_folder is not None:
        previous_history_path = Path(previous_simulation_folder, "history.xml")
        previous_measurements_path = Path(previous_simulation_folder, "measurements.xml")

    else:
        previous_history_path = None
        previous_measurements_path = None

    sim = Simulator.Simulator(op.simulation_folder, op.network_path, op.fleet_path, op.routes_path, op.sample_time,
                              previous_history_path, previous_measurements_path, op.std_factor, op.eta_model,
                              op.eta_table, op.soc_policy, op.start_earlier_by)
    return sim
