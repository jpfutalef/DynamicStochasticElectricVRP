import datetime
from pathlib import Path

import res.optimizer.alphaGA as alphaGA
from res.models import Fleet
from res.optimizer.GATools import AlphaGA_HyperParameters as HyperParameters


def pre_operation(fleet_filepath: Path, network_filepath: Path, soc_policy=(20, 95), additional_vehicles=1,
                  fill_up_to=0.9, external_individual=None):
    instance_name = network_filepath.stem
    instance_folder = network_filepath.parent
    now = datetime.datetime.now().strftime("%H-%M-%S_%d-%m-%y")
    multiple_results_folder = Path(instance_folder, instance_name)
    results_folder = Path(multiple_results_folder, now)

    fleet = Fleet.from_xml(fleet_filepath)
    network = Fleet.Network.from_xml(network_filepath)
    fleet.set_network(network)
    fleet.new_soc_policy(soc_policy[0], soc_policy[1])
    hp = HyperParameters(weights=(1., 1., 1., 0.),
                         num_individuals=4 * len(fleet) + 4 * len(fleet.network) + 20,
                         max_generations=5 * (4 * len(fleet) + 4 * len(fleet.network) + 20) + 40,
                         CXPB=0.5,
                         MUTPB=0.8,
                         hard_penalization=1000 * len(fleet.network),
                         elite_individuals=1,
                         tournament_size=5,
                         r=3,
                         alpha_up=fleet.vehicles[0].alpha_up)

    init_population = alphaGA.heuristic_population_3(hp.r, fleet, fill_up_to=fill_up_to,
                                                     additional_vehicles=additional_vehicles)
    if external_individual:
        init_population.insert(0, external_individual)
    result = alphaGA.alphaGA(fleet, hp, save_to=results_folder, init_pop=init_population)
    return result


def folder_pre_operation(folder_path: Path, repetitions: int = 5, soc_policy=(20, 95), additional_vehicles=1,
                         fill_up_to=0.9):
    # TODO Fix instance to fleet_path to network_path
    for instance in folder_path.iterdir():
        if instance.suffix == '.xml':
            for _ in range(repetitions):
                pre_operation(instance, soc_policy, additional_vehicles, fill_up_to)
