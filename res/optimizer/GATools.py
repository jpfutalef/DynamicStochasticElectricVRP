from pathlib import Path

from deap import base
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from typing import List, Tuple, Dict, NamedTuple, Union

import res.models.Fleet as Fleet
import res.tools.IOTools as IOTools

# TYPES
IndividualType = List
IndicesType = Dict[int, Tuple[int, int, int]]
RouteDict = Dict[int, Tuple[Tuple[int, ...], Tuple[float, ...], float, float, float]]


@dataclass
class HyperParameters:
    weights: Tuple[float, ...]
    num_individuals: int = 20
    max_generations: int = 40
    CXPB: float = 0.3
    MUTPB: float = 0.2
    hard_penalization: float = 100000
    elite_individuals: int = 1
    tournament_size: int = 5
    algorithm_name: str = 'Not specified'

    def __str__(self):
        string = 'Current hyper-parameters:\n'
        for key, val in self.__dict__.items():
            string += f'{key}:   {val}\n'
        return string

    def get_dataframe(self) -> pd.Series:
        return pd.Series(self.__dict__)

    def to_csv(self, filepath):
        self.get_dataframe().to_csv(filepath)


@dataclass
class AlphaGA_HyperParameters(HyperParameters):
    r: int = 2
    alpha_up: float = 95.
    crossover_repeat: int = 1
    mutation_repeat: int = 1
    algorithm_name: str = 'AlphaGA'


@dataclass
class BetaGA_HyperParameters(HyperParameters):
    algorithm_name: str = 'BetaGA'


@dataclass
class GenerationsData:
    generation: List = None
    best_fitness: List = None
    worst_fitness: List = None
    feasible: List = None
    average_fitness: List = None
    std_fitness: List = None

    def __post_init__(self):
        for key in self.__dict__.keys():
            self.__dict__[key] = []

    def save(self, data_folder):
        filepath = Path(data_folder, 'generations_data.csv')
        pd.DataFrame(self.__dict__, index=self.generation).to_csv(filepath)


@dataclass
class OptimizationData:
    fleet: Fleet.Fleet
    hyper_parameters: HyperParameters
    feasible: bool
    m: int
    algo_time: float = None
    best_fitness: float = None
    best_individual: List = None
    additional_info: Dict = None

    def save(self, folder_path):
        # Occupation file
        nodes_occupation_filepath = Path(folder_path, 'nodes_occupation.csv')
        theta_matrix = self.fleet.network.theta_matrix.T
        events = list(range(int(len(theta_matrix[:, 0]))))
        df_nodes_occupation = pd.DataFrame(theta_matrix, index=events)
        df_nodes_occupation.to_csv(nodes_occupation_filepath)

        # Fleet operation
        for id_ev, ev in self.fleet.vehicles.items():
            ev_filepath = Path(folder_path, f'EV{id_ev}_operation.csv')
            route_data = pd.DataFrame({'Sk': ev.S, 'Lk': ev.L})
            reaching_data = pd.DataFrame(ev.state_reaching.T, columns=['x1_reaching', 'x2_reaching', 'x3_reaching'])
            leaving_data = pd.DataFrame(ev.state_leaving.T, columns=['x1_leaving', 'x2_leaving', 'x3_leaving'])
            data = pd.concat([route_data, reaching_data, leaving_data], axis=1)
            data.to_csv(ev_filepath)

        # Routes
        routes_path = Path(folder_path, 'routes.xml')
        routes = {}
        departure_info = {}
        for id_ev, ev in self.fleet.vehicles.items():
            S, L = ev.S, ev.L
            w1 = (0.,)*len(ev.S)
            routes[id_ev] = (S, L, w1)
            departure_info[id_ev] = (ev.x1_0, ev.x2_0, ev.x3_0)
        IOTools.write_routes(routes_path, routes, departure_info)

        # Costs file
        cost_filepath = Path(folder_path, 'costs.csv')
        weight_tt, weight_ec, weight_chg_op, weight_chg_cost= self.hyper_parameters.weights
        cost_tt, cost_ec, cost_chg_op, cost_chg_cost = self.fleet.cost_function()
        index = ['weight', 'cost']
        data = [[weight_tt, weight_ec, weight_chg_op, weight_chg_cost],
                [cost_tt, cost_ec, cost_chg_op, cost_chg_cost]]
        df_costs = pd.DataFrame(data, columns=['Travel Time (min)', 'Energy Consumption (SOC)', 'Charging Time (min)',
                                               'Charging Cost'], index=index)
        df_costs.to_csv(cost_filepath)

        # Hyper-parameters file
        hyper_parameter_filepath = Path(folder_path, 'hyper_parameters.csv')
        self.hyper_parameters.to_csv(hyper_parameter_filepath)

        # Report file
        report_filepath = Path(folder_path, 'optimization_report.csv')
        report_data = {'Algorithm time': self.algo_time,
                       'm': self.m,
                       'feasible': self.feasible,
                       'best_fitness': self.best_fitness}
        if self.additional_info:
            report_data.update(self.additional_info)
        pd.Series(report_data).to_csv(report_filepath)

        # Save figures
        network_path = Path(folder_path, 'network')
        network_drawing_path = Path(folder_path, 'network.pdf')
        fig, g = self.fleet.network.draw(save_to=network_path, width=0.02,
                                         edge_color='grey')
        fig.savefig(network_drawing_path)

        figs = self.fleet.plot_operation_pyplot()
        for ev_id, fig in enumerate(figs[:-1]):
            png_figure_path = Path(folder_path, f'operation_EV{ev_id}')
            pdf_figure_path = Path(folder_path, f'operation_EV{ev_id}.pdf')
            fig.savefig(png_figure_path)
            fig.savefig(pdf_figure_path)
        png_occupation_figure_path = Path(folder_path, 'cs_occupation')
        pdf_occupation_figure_path = Path(folder_path, 'cs_occupation.pdf')
        figs[-1].savefig(png_occupation_figure_path)
        figs[-1].savefig(pdf_occupation_figure_path)

        fig, g = self.fleet.draw_operation(color_route=('r', 'b', 'g', 'c', 'y'), save_to=None, width=0.02,
                                           edge_color='grey')
        png_network_operation = Path(folder_path, 'network_operation')
        pdf_network_operation = Path(folder_path, 'network_operation.pdf')
        fig.savefig(png_network_operation)
        fig.savefig(pdf_network_operation)
        plt.close('all')

        # write best individual
        with open(Path(folder_path, 'best_individual.txt'), 'w') as file:
            file.write(str(self.best_individual))

        # write fleet
        fleet_path = Path(folder_path, 'fleet.xml')
        self.fleet.write_xml(fleet_path, network_in_file=False, print_pretty=False)
        # write network
        network_path = Path(folder_path, 'network.xml')
        self.fleet.network.write_xml(network_path, print_pretty=False)


class OptimizationReport(NamedTuple):
    best_fitness: float
    feasible: bool
    execution_time: float
    # hyper_parameters: Dict[str, float]


@dataclass
class FitnessMin(base.Fitness):
    weights: (-1.,)


@dataclass
class Individual(list):
    fitness: FitnessMin = FitnessMin()
    feasible: bool = False


@dataclass
class OptimizationDataParser:
    fleet: Fleet
    hyper_parameter: HyperParameters
    opt_data: GenerationsData


def save_optimization_report(path, report: OptimizationReport, print_pretty=False) -> None:
    tree = ET.parse(path)
    _info = tree.find('info')
    for _op_info in _info.findall('optimization_info'):
        _info.remove(_op_info)
    attrib = {'best_fitness': str(report.best_fitness),
              'feasible': str(report.feasible),
              'execution_time': str(report.execution_time)}
    _op_info = ET.SubElement(_info, 'optimization_info', attrib=attrib)

    if print_pretty:
        IOTools.write_pretty_xml(path, tree.getroot())
    else:
        tree.write(path)


def read_optimization_report(path, tree=None) -> Union[OptimizationReport, None]:
    if not tree:
        tree = ET.parse(path)
    _info = tree.find('info')
    _report = _info.find('optimization_info')
    if _report is not None:
        report = OptimizationReport(float(_report.get('best_fitness')), bool(_report.get('feasible')),
                                    float(_report.get('execution_time')))
        return report
    return None
