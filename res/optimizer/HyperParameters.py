from dataclasses import dataclass
from typing import Tuple
import pandas as pd


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
    a0: int = 20
    a1: float = 1.0
    a2: float = 1.0
    b0: int = 40
    b1: float = 1.0
    b2: float = 1.0

    def __str__(self):
        string = 'Current hyper-parameters:\n'
        for key, val in self.__dict__.items():
            string += f'        {key}:   {val}\n'
        return string

    def update_population_size(self, network_size: int, fleet_size: int):
        self.num_individuals = int(self.a2 * network_size + self.a1 * fleet_size + self.a0)

    def update_num_max_generations(self, network_size: int, fleet_size: int):
        self.max_generations = int(self.b2 * network_size + self.b1 * fleet_size + self.b0)

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
class OnGA_HyperParameters(HyperParameters):
    r: int = 2
    alpha_up: float = 95.
    crossover_repeat: int = 1
    mutation_repeat: int = 1
    algorithm_name: str = 'OnGA'
    offset_time_depot: float = 600.
