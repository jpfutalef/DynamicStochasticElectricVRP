"""
MAIN EXECUTION FILE
"""
import sys, os, argparse
from pathlib import Path
import numpy as np

from res.stages.PreOperation import folder_pre_operation
from res.stages.Online import online_operation, online_operation_degradation
from res.optimizer.GATools import OnGA_HyperParameters, AlphaGA_HyperParameters

operation_help = """Operation to perform.
(1) Pre-operation
(2) Online operation without optimization
(3) Online operation with optimization
(4) Online operation with optimization + degradation
"""

parser = argparse.ArgumentParser(description='Main file. You can execute all operational stages with it.')
parser.add_argument('operation', type=int, help=operation_help)
parser.add_argument('target_folder', type=Path, help='specifies working folder')

# pre-operation
parser.add_argument('--additional_vehicles', type=int, help='additional EVs in pre-operation. Default=0', default=0)
parser.add_argument('--fill_up_to', type=float, help='fill the EV with a mass up to this value. Default=1.0',
                    default=1.0)

# common options
parser.add_argument('--repetitions', type=int, help='Experiment repetitions. Default=50', default=50)

# online options
parser.add_argument('--sample_time', type=float, help='Online sample time in seconds. Default=300.0', default=300.0)
parser.add_argument('--keep_times', type=int, help='How many steps keep realizations. Default=0', default=0)
parser.add_argument('--std_factor', type=float, help='Noise gain. Default: 1.0', default=1.0)
parser.add_argument('--start_earlier_by', type=float,
                    help='Start the online operation earlier by this amount of seconds. Default: 0.0', default=0.0)

# degradation options
parser.add_argument('--soc_policy', type=int, nargs=2,
                    help='SOC policy considered by degradation simulation. Default: 20 95', default=[20, 95])

args = parser.parse_args()

hp = OnGA_HyperParameters(num_individuals=60,
                          max_generations=120,
                          CXPB=0.65,
                          MUTPB=0.8,
                          weights=(1., 1., 1., 0.),
                          r=2)

if __name__ == '__main__':
    # Pre-operation
    if args.operation == 1:
        instances_folder = args.target_folder
        if not instances_folder.is_dir():
            print("Directory is not valid: ", instances_folder)
            sys.exit(0)
        print("Will solve instances at:\n  ", instances_folder)
        input("Press any key to continue... (ctrl+Z to end process)")
        folder_pre_operation(instances_folder, repetitions=args.repetitions,
                             additional_vehicles=args.additional_vehicles, fill_up_to=args.fill_up_to)

    # Open loop
    elif args.operation == 2:
        source_folder = Path(args.target_folder, 'source')
        simulations_folder = Path(args.target_folder, 'simulations_OpenLoop')
        online_operation(args.target_folder, source_folder, False, hp, args.repetitions, args.keep_times,
                         args.sample_time, args.std_factor, args.start_earlier_by, args.soc_policy, False)

    # Closed loop
    elif args.operation == 3:
        source_folder = Path(args.target_folder, 'source')
        simulations_folder = Path(args.target_folder, 'simulations_ClosedLoop')
        online_operation(args.target_folder, source_folder, True, hp, args.repetitions, args.keep_times,
                         args.sample_time, args.std_factor, False)

    elif args.operation == 4:
        print(f'Considering the following policy:\n {args.soc_policy}')
        input('Press any key to continue... (ctrl+Z to finish process)')
        eta_table = np.array([[0.500, 1.00, 0.999930],
                              [0.625, 0.75, 0.999963],
                              [0.375, 0.75, 0.999972],
                              [0.750, 0.50, 0.999981],
                              [0.500, 0.50, 0.999987],
                              [0.250, 0.50, 0.999991],
                              [0.875, 0.25, 0.999997],
                              [0.625, 0.25, 0.999996],
                              [0.500, 0.25, 0.999996],
                              [0.375, 0.25, 0.999981],
                              [0.125, 0.25, 0.999987]])

        main_folder = args.target_folder
        source_folder = Path(main_folder, 'source')
        online_operation_degradation(source_folder, main_folder, hp, eta_table, keep_times=args.keep_times,
                                     sample_time=args.sample_time, std_factor=(1., 1.), policy=args.soc_policy,
                                     degrade_until=0.8)
    sys.exit(1)
