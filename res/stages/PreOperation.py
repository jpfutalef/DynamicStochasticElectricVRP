import datetime, multiprocessing
from pathlib import Path
from typing import Union, Type, Tuple

import pandas as pd

import res.optimizer.alphaGA as alphaGA
from res.models import Fleet
from res.models import Network
from res.optimizer.GATools import AlphaGA_HyperParameters


def get_best_from_pre_operation(folder: Path) -> Path:
    data = []
    for results_folder in folder.iterdir():
        try:
            report_df = pd.read_csv(Path(results_folder, 'optimization_report.csv'), index_col=0)
            to_add = [float(report_df.loc['best_fitness'][0]), results_folder]
            data.append(to_add)
        except FileNotFoundError:
            pass
    df = pd.DataFrame(data, columns=['fitness', 'folder'])
    df = df.sort_values('fitness')
    return df.iloc[0]['folder']


def pre_operation(instance_filepath: Path, hp: AlphaGA_HyperParameters, soc_policy: Tuple[float, float] = None,
                  additional_vehicles=1, fill_up_to=0.9, external_individual=None, sat_prob_sample_time: float = 120,
                  cs_capacities: int = None, method_name: str = None, repetitions: int = 5,
                  fleet_type: Union[Type[Fleet.Fleet], str] = None,
                  ev_type: Union[Type[Fleet.EV.ElectricVehicle], str] = None,
                  network_type: Union[Type[Network.Network], str] = None,
                  edge_type: Union[Type[Network.Edge.DynamicEdge], str] = None):
    # Setup directories
    if method_name:
        multiple_results_folder = Path(instance_filepath.parent, instance_filepath.stem, method_name, 'pre_operation')
    else:
        multiple_results_folder = Path(instance_filepath.parent, instance_filepath.stem, 'pre_operation')

    # An empty list that will contain multiple optimization results
    results = []

    # Iterate repetitions
    for _ in range(repetitions):
        # This execution directory
        now = datetime.datetime.now().strftime("%H-%M-%S_%d-%m-%y")
        execution_folder = Path(multiple_results_folder, now)

        # Read fleet and network from XML files independently
        fleet = Fleet.from_xml(instance_filepath, fleet_type, ev_type)
        network = Network.from_xml(instance_filepath, network_type, edge_type)

        # Set network to fleet
        fleet.set_network(network)

        # Setup fleet and network
        if type(fleet) is Fleet.GaussianFleet:
            fleet.set_saturation_probability_sample_time(sat_prob_sample_time)
        if cs_capacities:
            fleet.modify_cs_capacities(cs_capacities)
        if soc_policy:
            fleet.new_soc_policy(soc_policy[0], soc_policy[1])

        # If warm start solution, use it. Otherwise, create individual using heuristic
        if external_individual:
            init_population = [external_individual]
            fleet_size = external_individual.count('|')
            fleet.resize_fleet(fleet_size)
        else:
            init_population = alphaGA.heuristic_population_3(hp.r, fleet, fill_up_to=fill_up_to,
                                                             additional_vehicles=additional_vehicles)

        # Execute GA
        result = alphaGA.alphaGA(fleet, hp, save_to=execution_folder, init_pop=init_population,
                                 instance_path=instance_filepath)

        # Append result to list
        results.append(result)

    # write file with directory of best result
    s = get_best_from_pre_operation(multiple_results_folder)
    with open(Path(multiple_results_folder, 'source_folder.txt'), 'w') as file:
        file.write(str(s))

    return results, multiple_results_folder


def folder_pre_operation(folder_path: Path, hp: AlphaGA_HyperParameters, soc_policy: Tuple[float, float] = None,
                         additional_vehicles=1, fill_up_to=0.9, external_individual=None, method_name: str = None,
                         repetitions: int = 5, fleet_type: Union[Type[Fleet.Fleet], str] = None,
                         ev_type: Union[Type[Fleet.EV.ElectricVehicle], str] = None,
                         network_type: Union[Type[Fleet.Network.Network], str] = None,
                         edge_type: Union[Type[Fleet.Network.Edge.DynamicEdge], str] = None,
                         sat_prob_sample_time: float = 120, cs_capacities: int = None, parallel=True):
    instances = [i for i in folder_path.iterdir() if i.suffix == '.xml']
    other_args = (hp, soc_policy, additional_vehicles, fill_up_to, external_individual, sat_prob_sample_time,
                  cs_capacities, method_name, repetitions, fleet_type, ev_type, network_type, edge_type)
    if parallel:
        num_cores = multiprocessing.cpu_count()
        pool_size = len(instances) if len(instances) < num_cores else num_cores
        args = [(i,) + other_args for i in instances]
        pool = multiprocessing.Pool(processes=pool_size)
        pool.starmap_async(pre_operation, args)
        pool.close()
        pool.join()

    else:
        for instance in instances:
            pre_operation(instance, hp, soc_policy, additional_vehicles, fill_up_to, external_individual,
                          sat_prob_sample_time, cs_capacities, method_name, repetitions, fleet_type,
                          ev_type, network_type, edge_type)
