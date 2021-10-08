from pathlib import Path
import pandas as pd
from typing import Type, Union, Tuple

from res.stages import PreOperation
from res.stages import Online
from res.models import Fleet
from res.optimizer import GATools
from res.tools import multiprocess


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


def one_day(instance_filepath: Path, soc_policy: Tuple[float, float] = None, additional_vehicles=1,
            fill_up_to=0.9, external_individual=None, method_name: str = None, pre_operation_repetitions: int = 5,
            online_sample_time: float = 300, std_factor: float = 1.0, start_earlier_by: float = 600,
            sat_prob_sample_time: float = 120, cs_capacities: int = None,
            online_repetitions: int = 50, fleet_type: Union[Type[Fleet.Fleet], str] = None,
            ev_type: Union[Type[Fleet.EV.ElectricVehicle], str] = None,
            network_type: Union[Type[Fleet.Network.Network], str] = None,
            edge_type: Union[Type[Fleet.Network.Edge.DynamicEdge], str] = None,
            optimize_online: bool = False, onGA_hp: GATools.OnGA_HyperParameters = None,
            parallel=True):
    # Pre-operation
    potential_preop_path = Path(instance_filepath.parent, instance_filepath.stem, method_name, 'pre_operation')
    if potential_preop_path.is_dir():
        pre_operation_folder = potential_preop_path
    else:
        _, pre_operation_folder = PreOperation.pre_operation(instance_filepath, soc_policy, additional_vehicles,
                                                             fill_up_to, external_individual, sat_prob_sample_time,
                                                             cs_capacities, method_name, pre_operation_repetitions,
                                                             fleet_type, ev_type, network_type, edge_type)

    # Select folder with best result and save it to text file
    source_folder = get_best_from_pre_operation(pre_operation_folder)

    # Online
    if parallel:
        args = [(pre_operation_folder.parent, source_folder, True, onGA_hp, online_repetitions,
                 0, online_sample_time, std_factor, start_earlier_by, soc_policy, False, ev_type,
                 fleet_type, edge_type, network_type),
                (pre_operation_folder.parent, source_folder, False, onGA_hp, online_repetitions,
                 0, online_sample_time, std_factor, start_earlier_by, soc_policy, False, ev_type,
                 fleet_type, edge_type, network_type)]
        pool = multiprocess.MyPool(processes=2)
        pool.starmap_async(Online.online_operation, args)
        pool.close()
        pool.join()

        # p1 = multiprocess.multiprocessing.Process(target=Online.online_operation, args=args[0])
        # p2 = multiprocess.multiprocessing.Process(target=Online.online_operation, args=args[1])
        #
        # p1.start()
        # p2.start()
        #
        # p1.close()
        # p1.join()
        #
        # p2.close()
        # p2.join()
    else:
        Online.online_operation(pre_operation_folder.parent, source_folder, False, onGA_hp, online_repetitions,
                                0, online_sample_time, std_factor, start_earlier_by, soc_policy, False, ev_type,
                                fleet_type, edge_type, network_type)
        Online.online_operation(pre_operation_folder.parent, source_folder, True, onGA_hp, online_repetitions,
                                0, online_sample_time, std_factor, start_earlier_by, soc_policy, False, ev_type,
                                fleet_type, edge_type, network_type)
    return


def one_day_folder(folder_path: Path, soc_policy: Tuple[float, float] = None, additional_vehicles=1,
                   fill_up_to=0.9, external_individual=None, method_name: str = None,
                   pre_operation_repetitions: int = 5, online_sample_time: float = 600, std_factor: float = 1.0,
                   start_earlier_by: float = 600,
                   sat_prob_sample_time: float = 120, cs_capacities: int = None,
                   online_repetitions: int = 50, fleet_type: Union[Type[Fleet.Fleet], str] = None,
                   ev_type: Union[Type[Fleet.EV.ElectricVehicle], str] = None,
                   network_type: Union[Type[Fleet.Network.Network], str] = None,
                   edge_type: Union[Type[Fleet.Network.Edge.DynamicEdge], str] = None,
                   optimize_online: bool = False, onGA_hp: GATools.OnGA_HyperParameters = None,
                   parallel=True):
    instances = [i for i in folder_path.iterdir() if i.suffix == '.xml']
    other_args = (soc_policy, additional_vehicles, fill_up_to, external_individual, method_name,
                  pre_operation_repetitions, online_sample_time, std_factor, start_earlier_by, sat_prob_sample_time,
                  cs_capacities, online_repetitions, fleet_type, ev_type, network_type, edge_type, optimize_online,
                  onGA_hp, parallel)

    if parallel:
        pool_size = len(instances)
        args = [(i,) + other_args for i in instances]
        pool = multiprocess.MyPool(processes=pool_size)
        pool.starmap_async(one_day, args)
        pool.close()
        pool.join()

    else:
        for instance in [i for i in folder_path.iterdir() if i.suffix == '.xml']:
            one_day(instance, soc_policy, additional_vehicles, fill_up_to, external_individual, method_name,
                    pre_operation_repetitions, online_sample_time, std_factor, start_earlier_by, sat_prob_sample_time,
                    cs_capacities, online_repetitions, fleet_type, ev_type, network_type, edge_type, optimize_online,
                    onGA_hp, parallel)
