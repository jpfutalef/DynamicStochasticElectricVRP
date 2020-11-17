import datetime
import os

from deap import base
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from typing import List, Tuple, Dict, NamedTuple, Union

from models.Fleet import Fleet, InitialCondition
from tools.IOTools import write_pretty_xml

# TYPES
IndividualType = List
IndicesType = Dict[int, Tuple[int, int]]
StartingPointsType = Dict[int, InitialCondition]
RouteVector = Tuple[Tuple[int, ...], Tuple[float, ...]]
RouteDict = Dict[int, Tuple[RouteVector, float, float, float]]

'''
class FitnessMin(base.Fitness):
    weights = (-1.0,)


class Individual(list):
    feasible = False
    acceptable = False

    def __init__(self):
        super().__init__()
        self.fitness = FitnessMin()

    def __getitem__(self, key):
        return super(Individual, self).__getitem__(key - 1)
        
'''


@dataclass
class HyperParameters:
    num_individuals: int
    max_generations: int
    CXPB: float
    MUTPB: float
    weights: Tuple[float, ...]
    K1: float
    K2: float
    keep_best: int
    tournament_size: int
    r: int = 2
    alpha_up: float = 80.
    algorithm_name: str = 'Not specified'
    crossover_repeat: int = 1
    mutation_repeat: int = 1

    def __str__(self):
        string = 'Current hyper-parameters:\n'
        for key, val in self.__dict__.items():
            string += f'{key}:   {val}\n'
        return string

    def get_dataframe(self):
        return pd.Series(self.__dict__)


@dataclass
class GenerationsData:
    generations: List
    best_fitness: List
    worst_fitness: List
    average_fitness: List
    std_fitness: List
    best_individuals: List
    fleet: Fleet
    hyper_parameters: HyperParameters
    bestOfAll: List
    feasible: bool
    acceptable: bool
    m: int
    cs_capacity: int
    algo_time: float = None
    additional_info: Dict = None

    def save_opt_data(self, data_folder: str = None, method='ASSIGNATION', savefig=False):
        # folder
        if data_folder:
            opt_path = data_folder
        else:
            now = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
            folder_name = f'{now}_FEASIBLE_{method}/' if self.feasible else f'{now}_INFEASIBLE_{method}/'
            opt_path = data_folder + folder_name

            try:
                os.mkdir(opt_path)
            except FileExistsError:
                pass

        # files
        optimization_iterations_filepath = opt_path + 'optimization_iterations.csv'
        theta_vector_filepath = opt_path + 'nodes_occupation.csv'
        info_filepath = opt_path + 'hyper-parameters.csv'

        # optimization results
        df_op_gens = pd.DataFrame({'best_fitness': self.best_fitness, 'worst_fitness': self.worst_fitness,
                                   'pop_average': self.average_fitness, 'pop_std': self.std_fitness},
                                  index=self.generations)
        df_op_gens.to_csv(optimization_iterations_filepath)

        # theta vector
        mt = self.fleet.theta_matrix.T
        events = list(range(int(len(mt[:, 0]))))
        df_nodes_occupation = pd.DataFrame(mt, index=events)
        df_nodes_occupation.to_csv(theta_vector_filepath)

        # fleet operation
        for id_ev, ev in self.fleet.vehicles.items():
            ev_filepath = opt_path + f'EV{id_ev}_operation.csv'
            route_data = pd.DataFrame({'Sk': ev.route[0], 'Lk': ev.route[1]})
            reaching_data = pd.DataFrame(ev.state_reaching.T, columns=['x1_reaching', 'x2_reaching', 'x3_reaching'])
            leaving_data = pd.DataFrame(ev.state_leaving.T, columns=['x1_leaving', 'x2_leaving', 'x3_leaving'])
            wt0 = pd.DataFrame(ev.waiting_times0.T, columns=['wating_time_after'])
            wt1 = pd.DataFrame(ev.waiting_times1.T, columns=['wating_time_before'])
            data = pd.concat([route_data, reaching_data, leaving_data, wt0, wt1], axis=1)
            data.to_csv(ev_filepath)

        # costs
        cost_filepath = opt_path + 'costs.csv'
        weight_tt, weight_ec, weight_chg_op, weight_chg_cost, weight_wait_time = self.hyper_parameters.weights
        cost_tt, cost_ec, cost_chg_op, cost_chg_cost, cost_weight_time = self.fleet.cost_function()
        index = ['weight', 'cost']
        data = [[weight_tt, weight_ec, weight_chg_op, weight_chg_cost, weight_wait_time],
                [cost_tt, cost_ec, cost_chg_op, cost_chg_cost, cost_weight_time]]
        df_costs = pd.DataFrame(data, columns=['Travel Time (min)', 'Energy Consumption (SOC)',
                                               'Charging Time (min)', 'Charging Cost', 'Waiting Time (min)'],
                                index=index)
        df_costs.to_csv(cost_filepath)

        # save hyper-parameters
        info_df = self.hyper_parameters.get_dataframe()
        additional_info = {'Algorithm time': self.algo_time,
                           'Best individual': [i for i in self.bestOfAll],
                           'm': self.m,
                           'cs_capacity': self.cs_capacity,
                           'feasible': self.feasible,
                           'acceptable': self.acceptable}
        if self.additional_info:
            additional_info.update(self.additional_info)
        info_df = info_df.append(pd.Series(additional_info))
        info_df.to_csv(info_filepath)


        # Edit assignation file
        self.fleet.assign_customers_in_route()
        assigned_path = f'{opt_path}result_instance.xml'
        self.fleet.write_xml(assigned_path, True, True, True, False, False)

        if savefig:
            fig, g = self.fleet.network.draw(save_to=f'{opt_path}network', width=0.02,
                                             edge_color='grey',
                                             markeredgecolor='black', markeredgewidth=2.0)
            fig.savefig(f'{opt_path}network.pdf')

            figs = self.fleet.plot_operation_pyplot()
            for ev_id, fig in enumerate(figs[:-1]):
                fig.savefig(f'{opt_path}operation_EV{ev_id}')
                fig.savefig(f'{opt_path}operation_EV{ev_id}.pdf')
            figs[-1].savefig(f'{opt_path}cs_occupation')
            figs[-1].savefig(f'{opt_path}cs_occupation.pdf')

            fig, g = self.fleet.draw_operation(color_route=('r', 'b', 'g', 'c', 'y'), save_to=None, width=0.02,
                                               edge_color='grey',
                                               markeredgecolor='black', markeredgewidth=2.0)

            fig.savefig(f'{opt_path}network_operation')
            fig.savefig(f'{opt_path}network_operation.pdf')
            plt.close('all')


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


def save_optimization_report(path, report: OptimizationReport, pretty=False) -> None:
    tree = ET.parse(path)
    _info = tree.find('info')
    for _op_info in _info.findall('optimization_info'):
        _info.remove(_op_info)
    attrib = {'best_fitness': str(report.best_fitness),
              'feasible': str(report.feasible),
              'execution_time': str(report.execution_time)}
    _op_info = ET.SubElement(_info, 'optimization_info', attrib=attrib)

    tree.write(path)
    if pretty:
        write_pretty_xml(path)
    return


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
