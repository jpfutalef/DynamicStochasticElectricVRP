import datetime
from pathlib import Path
from typing import Union, Type, Tuple

import res.optimizer.alphaGA as alphaGA
from res.models import Fleet
from res.optimizer.GATools import AlphaGA_HyperParameters as HyperParameters


def pre_operation(fleet_filepath: Union[Path, None], network_filepath: Union[Path, None],
                  instance_filepath: Path = None, soc_policy: Tuple[float, float] = None,
                  additional_vehicles=1, fill_up_to=0.9, external_individual=None,
                  fleet_type: Union[Type[Fleet.Fleet], str] = None,
                  ev_type: Union[Type[Fleet.EV.ElectricVehicle], str] = None,
                  network_type: Union[Type[Fleet.Network.Network], str] = None,
                  edge_type: Union[Type[Fleet.Network.Edge.DynamicEdge], str] = None,
                  sat_prob_sample_time: float = 120, cs_capacities: int = None,
                  results_folder_suffix: str = None):
    if instance_filepath:
        instance_name = instance_filepath.stem
        instance_folder = instance_filepath.parent

        fleet = fleet_type.from_xml(instance_filepath, ev_type=ev_type)
        network = network_type.from_xml(instance_filepath, edge_type=edge_type)

    else:
        instance_name = network_filepath.stem
        instance_folder = network_filepath.parent

        fleet = fleet_type.from_xml(fleet_filepath, ev_type=ev_type)
        network = network_type.from_xml(network_filepath, edge_type=edge_type)

    fleet.set_network(network)

    multiple_results_folder = Path(instance_folder, instance_name)
    now = datetime.datetime.now().strftime("%H-%M-%S_%d-%m-%y")

    if results_folder_suffix is None:
        results_folder = Path(multiple_results_folder, now)
    else:
        results_folder = Path(multiple_results_folder, results_folder_suffix, now)

    if type(fleet) is Fleet.GaussianFleet:
        fleet.set_saturation_probability_sample_time(sat_prob_sample_time)
    if cs_capacities:
        fleet.modify_cs_capacities(cs_capacities)
    if soc_policy:
        fleet.new_soc_policy(soc_policy[0], soc_policy[1])

    hp = HyperParameters(weights=(1., 1., 1., 0.),
                         num_individuals=4 * len(fleet) + 4 * len(fleet.network) + 20,
                         max_generations=4 * (4 * len(fleet) + 4 * len(fleet.network) + 20) + 40,
                         CXPB=0.5,
                         MUTPB=0.8,
                         hard_penalization=1000 * len(fleet.network),
                         elite_individuals=1,
                         tournament_size=5,
                         r=2,
                         alpha_up=fleet.vehicles[0].alpha_up)

    if external_individual:
        init_population = [external_individual]
        fleet_size = external_individual.count('|')
        fleet.resize_fleet(fleet_size)
    else:
        init_population = alphaGA.heuristic_population_3(hp.r, fleet, fill_up_to=fill_up_to,
                                                         additional_vehicles=additional_vehicles)
    result = alphaGA.alphaGA(fleet, hp, save_to=results_folder, init_pop=init_population)
    return result


def folder_pre_operation(folder_path: Path = None, repetitions: int = 5,
                         soc_policy: Tuple[float, float] = None,
                         additional_vehicles=1, fill_up_to=0.9, external_individual=None,
                         fleet_type: Union[Type[Fleet.Fleet], str] = None,
                         ev_type: Union[Type[Fleet.EV.ElectricVehicle], str] = None,
                         network_type: Union[Type[Fleet.Network.Network], str] = None,
                         edge_type: Union[Type[Fleet.Network.Edge.DynamicEdge], str] = None,
                         sat_prob_sample_time: float = 120, cs_capacities: int = None,
                         results_folder_suffix: str = None):
    for instance in folder_path.iterdir():
        if instance.suffix == '.xml':
            for _ in range(repetitions):
                pre_operation(None, None, instance, soc_policy, additional_vehicles, fill_up_to, external_individual,
                              fleet_type, ev_type, network_type, edge_type, sat_prob_sample_time, cs_capacities,
                              results_folder_suffix)
