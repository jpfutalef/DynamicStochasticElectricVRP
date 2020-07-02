from models.Fleet import *
from typing import Tuple
from deap import base, creator, tools
import datetime, os
import pandas as pd


class HyperParameters:
    def __init__(self, num_individuals: int, max_generations: int, CXPB: float, MUTPB: float,
                 weights: Tuple[float, ...], penalization_constant: float, keep_best: int, **kwargs):
        self.num_individuals = num_individuals
        self.max_generations = max_generations
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.weights = weights
        self.penalization_constant = penalization_constant
        self.keep_best = keep_best
        for key, val in kwargs.items():
            self.__dict__[key] = val

    def __str__(self):
        string = 'Current hyper-parameters:\n'
        for key, val in self.__dict__.items():
            string += f'  {key}:   {val}\n'
        return string


@dataclass
class OptimizationIterationsData:
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
    algo_time: float = None

    def save_opt_data(self, data_folder: str):
        # folder
        now = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        folder_name = f'{now}_FEASIBLE_ASSIGNATION' if self.feasible else f'{now}_INFEASIBLE_ASSIGNATION/'
        opt_path = data_folder + folder_name

        try:
            os.mkdir(opt_path)
        except FileExistsError:
            pass

        # files
        optimization_iterations_filepath = opt_path + '/optimization_iterations.csv'
        theta_vector_filepath = opt_path + '/nodes_occupation.csv'
        info_filepath = opt_path + '/hyper-parameters.txt'

        # optimization results
        df_op_gens = pd.DataFrame({'best_fitness': self.best_fitness, 'worst_fitness': self.worst_fitness,
                                   'pop_average': self.average_fitness, 'pop_std': self.std_fitness},
                                  index=self.generations)
        df_op_gens.to_csv(optimization_iterations_filepath)

        # theta vector
        theta_vector = self.fleet.optimization_vector[self.fleet.optimization_vector_indices[8]:]
        net_size = len(self.fleet.network.nodes)
        events = list(range(int(len(theta_vector) / net_size)))
        theta_matrix = np.array([theta_vector[i * net_size:net_size * (i + 1)] for i in events])
        df_nodes_occupation = pd.DataFrame(theta_matrix, index=events)
        df_nodes_occupation.to_csv(theta_vector_filepath)

        # fleet operation
        for id_ev, ev in self.fleet.vehicles.items():
            ev_filepath = opt_path + f'/EV{id_ev}_operation.csv'
            reaching_data = pd.DataFrame(ev.state_reaching.T, columns=['x1_reaching', 'x2_reaching', 'x3_reaching'])
            leaving_data = pd.DataFrame(ev.state_leaving.T, columns=['x1_leaving', 'x2_leaving', 'x3_leaving'])
            route_data = pd.DataFrame({'Sk': ev.route[0], 'Lk': ev.route[1]})
            data = pd.concat([route_data, reaching_data, leaving_data], axis=1)
            data.to_csv(ev_filepath)

        # costs
        cost_filepath = opt_path + '/costs.csv'
        weight_tt, weight_ec, weight_chg_op, weight_chg_cost = self.hyper_parameters.weights
        cost_tt, cost_ec, cost_chg_op, cost_chg_cost = self.fleet.cost_function()
        index = ['weight', 'cost']
        data = [[weight_tt, weight_ec, weight_chg_op, weight_chg_cost],
                [cost_tt, cost_ec, cost_chg_op, cost_chg_cost]]
        df_costs = pd.DataFrame(data, columns=['Travel Time (min)', 'Energy Consumption (SOC)',
                                               'Charging Time (min)', 'Charging Cost'], index=index)
        df_costs.to_csv(cost_filepath)

        # save hyper-parameters
        info = self.hyper_parameters.__str__()
        info += f'Algorithm Time: {self.algo_time}'
        info += f'\nBest individual: {self.bestOfAll}'
        with open(info_filepath, 'w') as file:
            file.write(info)

        # Edit assignation file
        self.fleet.assign_customers_in_route()
        assigned_path = f'{opt_path}/assigned.xml'
        self.fleet.write_xml(assigned_path, True, True, False, True)

        '''
        tree = ET.parse(assigned_path)
        _fleet = tree.find('fleet')

        for _ev, ev in zip(_fleet, fleet.vehicles.values()):
            while _ev.find('assigned_customers'):
                _ev.remove(_ev.find('assigned_customers'))
            _assigned_customers = ET.SubElement(_ev, 'assigned_customers')
            for node in ev.assigned_customers:
                _node = ET.SubElement(_assigned_customers, 'node', attrib={'id': str(node)})

        tree.write(assigned_path)
        '''


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
    opt_data: OptimizationIterationsData


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


def write_pretty_xml(path):
    xml_pretty = xml.dom.minidom.parse(path).toprettyxml()
    with open(path, 'w') as file:
        file.write(xml_pretty)


def fitness(routes: RouteDict, fleet: Fleet, hyper_parameters: HyperParameters):
    # weights and penalization constant
    weights, penalization_constant = hyper_parameters.weights, hyper_parameters.penalization_constant

    # Set routes
    fleet.set_routes_of_vehicles(routes)

    # Get optimization vector
    fleet.create_optimization_vector()

    # Cost
    costs = fleet.cost_function()

    # Check if the solution is feasible
    feasible, penalization = fleet.feasible()

    # penalization
    if not feasible:
        penalization += penalization_constant

    # calculate and return
    fit = np.dot(np.asarray(costs), np.asarray(weights)) + penalization
    return fit, feasible

