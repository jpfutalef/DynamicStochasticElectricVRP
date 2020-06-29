from Fleet import *
from typing import Tuple
from deap import base, creator, tools


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
    statistics: Dict[str, list]