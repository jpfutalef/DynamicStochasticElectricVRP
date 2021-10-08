from res.tools import multiprocess
import os
import multiprocessing as mp
from time import sleep

'''
This list runs full days
'''
experiments1 = ["3 1 data/instances/set0/ --preop_repetitions 2 --online_repetitions 2 --parallel",
                "3 2 data/instances/set0/ --preop_repetitions 2 --online_repetitions 2 --parallel"]

# Parameters
experiments = experiments1
parallel = True # Doesn't work yet

'''
CODE
'''


def experiment(experiment_description: str):
    command = f'python main.py {experiment_description}'
    os.system(command)


def run_experiments(l, in_parallel=False):
    if in_parallel:
        pass
    else:
        [experiment(i) for i in l]


if __name__ == '__main__':
    run_experiments(experiments, parallel)
