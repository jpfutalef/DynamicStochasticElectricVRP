import sys, os, argparse
from pathlib import Path
import numpy as np

from res.stages import PreOperation, Online, OneDay
from res.optimizer.HyperParameters import AlphaGA_HyperParameters, OnGA_HyperParameters
import res.models.Fleet as Fleet
import res.models.Network as Network

"""
HYPER-PARAMETERS DEFINITIONS
"""
off_hp = AlphaGA_HyperParameters(weights=(1., 1., 1., 0.),
                                 num_individuals=20,    # Not considered as a0, a1, b0, b1 are assigned
                                 max_generations=40,
                                 CXPB=0.6,
                                 MUTPB=0.4,
                                 hard_penalization=10000,
                                 elite_individuals=1,
                                 tournament_size=5,
                                 algorithm_name='OffGA',
                                 a0=20,
                                 a1=4.0,
                                 a2=3.0,
                                 b0=30,
                                 b1=8.0,
                                 b2=6.0,
                                 r=2)

on_hp = OnGA_HyperParameters(num_individuals=60,
                             max_generations=120,
                             CXPB=0.65,
                             MUTPB=0.8,
                             weights=(1., 1., 1., 0.),
                             r=2)
"""
END OF HYPER-PARAMETERS DEFINITION
"""

stage_help = """Stage to study.
(1) Pre-operation
(2) Online
(3) One day
"""

variant_help = """E-VRP variant considered by the optimizer.
(1) Deterministic
(2) Deterministic with waiting times
(3) Linear stochastic
Default: 1
"""

# create parser instance and mandatory arguments
parser = argparse.ArgumentParser(description='Main file. You can execute all operational stages with it.')
parser.add_argument('stage', type=int, help=stage_help)
parser.add_argument('target_folder', type=Path, help='folder where target instances are located')

# common options
parser.add_argument('--variant', type=int, help=variant_help, default=1)
parser.add_argument('--cs_capacities', type=int, help='Capacity of CSs. Default=2', default=2)
parser.add_argument('--preop_repetitions', type=int, help='Num. of pre-operation runs. Default=5', default=5)
parser.add_argument('--sat_prob_sample_time', type=int,
                    help='Sample time of CS capacity saturation probability. Default=120 (2 min)', default=120)

# pre-operation options
parser.add_argument('--additional_vehicles', type=int, help='additional EVs in pre-operation. Default=0', default=0)
parser.add_argument('--fill_up_to', type=float, help='fill the EV with a mass up to this value. Default=1.0',
                    default=1.0)

# online options
parser.add_argument('--optimize', action='store_true', help='If passed, activates closed-loop operation.')
parser.add_argument('--source_folder', type=Path,
                    help='Source folder containing pre-operation results. Must be passed for simulations Default=None',
                    default=None)
parser.add_argument('--sample_time', type=float, help='Online sample time in seconds. Default=300.0', default=300.0)
parser.add_argument('--keep_times', type=int, help='How many steps keep realizations. Default=0', default=0)
parser.add_argument('--std_factor', type=float, help='Noise gain. Default: 1.0', default=1.0)
parser.add_argument('--start_earlier_by', type=float,
                    help='Start the simulation earlier by this amount of seconds. Default: 1200.0 [s] (20 min)',
                    default=1200.0)
parser.add_argument('--online_repetitions', type=int, help='Number of online stage simulations', default=20)

# degradation options
parser.add_argument('--soc_policy', type=int, nargs=2,
                    help='SOC policy considered by degradation simulation. Default: 20 95', default=[20, 95])

# Miscelaneous options
parser.add_argument('--display_gui', action='store_true', help='Display GUI from methods/stages/etc (if any)')
parser.add_argument('--parallel', action='store_true', help='Enables parallelism')

args = parser.parse_args()

method = 'deterministic'
ev_type = Fleet.EV.ElectricVehicle
fleet_type = Fleet.Fleet
network_type = Network.DeterministicCapacitatedNetwork
edge_type = Network.Edge.DynamicEdge

if args.variant == 2:
    method = 'deterministic_waiting_times'
    ev_type = Fleet.EV.ElectricVehicleWithWaitingTimes

elif args.variant == 3:
    method = 'stochastic_linear'
    ev_type = Fleet.EV.GaussianElectricVehicle
    fleet_type = Fleet.GaussianFleet
    edge_type = Network.Edge.GaussianEdge
    network_type = Network.GaussianCapacitatedNetwork

if __name__ == '__main__':
    # Pre-operation stage
    if args.stage == 1:
        instances_folder = args.target_folder
        if not instances_folder.is_dir():
            print("Directory is not valid: ", instances_folder)
            sys.exit(0)
        print("Will solve instances at:\n  ", instances_folder)
        input("Press ENTER to continue... (or ctrl+Z to end process)")
        PreOperation.folder_pre_operation(args.target_folder, off_hp, args.soc_policy, args.additional_vehicles,
                                          args.fill_up_to, None, method, args.preop_repetitions, fleet_type,
                                          ev_type, network_type, edge_type, args.sat_prob_sample_time,
                                          args.cs_capacities, args.parallel)

    # Online stage
    elif args.stage == 2:
        source_folder = args.target_folder
        Online.online_operation(source_folder.parent.parent, source_folder, args.optimize, on_hp,
                                args.online_repetitions,
                                args.keep_times, args.sample_time, args.std_factor, args.start_earlier_by,
                                args.soc_policy, False, ev_type, fleet_type, edge_type, network_type)

    # Complete day
    elif args.stage == 3:
        OneDay.one_day_folder(args.target_folder, args.soc_policy, args.additional_vehicles, args.fill_up_to, None,
                              method, args.preop_repetitions, args.sample_time, args.std_factor, args.start_earlier_by,
                              args.sat_prob_sample_time, args.cs_capacities, args.online_repetitions, fleet_type,
                              ev_type, network_type, edge_type, args.optimize, on_hp, args.parallel)

    # Closed loop with degradation
    elif args.case == 500:
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
        Online.online_operation_degradation(source_folder, main_folder, on_hp, eta_table, keep_times=args.keep_times,
                                            sample_time=args.sample_time, std_factor=(1., 1.), policy=args.soc_policy,
                                            degrade_until=0.8)
    sys.exit(1)
