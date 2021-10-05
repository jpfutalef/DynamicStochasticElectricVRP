from pathlib import Path
import pandas as pd
from typing import Type, Union, Tuple

from res.stages import PreOperation
from res.stages import Online
from res.models import Fleet
from res.optimizer import GATools


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
            sat_prob_sample_time: float = 120, cs_capacities: int = None,
            online_repetitions: int = 50, fleet_type: Union[Type[Fleet.Fleet], str] = None,
            ev_type: Union[Type[Fleet.EV.ElectricVehicle], str] = None,
            network_type: Union[Type[Fleet.Network.Network], str] = None,
            edge_type: Union[Type[Fleet.Network.Edge.DynamicEdge], str] = None,
            optimize_online: bool = False, onGA_hp: GATools.OnGA_HyperParameters = None):
    # Pre-operation
    _, pre_operation_folder = PreOperation.pre_operation(instance_filepath, soc_policy, additional_vehicles,
                                                         fill_up_to, external_individual, sat_prob_sample_time,
                                                         cs_capacities, method_name, pre_operation_repetitions,
                                                         fleet_type, ev_type, network_type, edge_type)

    # Select folder with best result and save it to text file
    source_folder = get_best_from_pre_operation(pre_operation_folder)

    # Online
    Online.online_operation(pre_operation_folder, source_folder, optimize_online, onGA_hp, online_repetitions)

    return


def one_day_folder(folder: Path, soc_policy: Tuple[float, float] = None, additional_vehicles=1,
                   fill_up_to=0.9, external_individual=None, method_name: str = None,
                   pre_operation_repetitions: int = 5,
                   sat_prob_sample_time: float = 120, cs_capacities: int = None,
                   online_repetitions: int = 50, fleet_type: Union[Type[Fleet.Fleet], str] = None,
                   ev_type: Union[Type[Fleet.EV.ElectricVehicle], str] = None,
                   network_type: Union[Type[Fleet.Network.Network], str] = None,
                   edge_type: Union[Type[Fleet.Network.Edge.DynamicEdge], str] = None,
                   optimize_online: bool = False, onGA_hp: GATools.OnGA_HyperParameters = None):
    for instance in [i for i in folder.iterdir() if i.suffix == '.xml']:
        one_day(instance, soc_policy, additional_vehicles, fill_up_to, external_individual, method_name,
                pre_operation_repetitions, sat_prob_sample_time, cs_capacities,  online_repetitions, fleet_type,
                ev_type, network_type, edge_type, optimize_online, onGA_hp)
