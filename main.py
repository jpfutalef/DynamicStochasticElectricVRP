"""
MAIN EXECUTION FILE
"""
import sys, os, argparse
from pathlib import Path
import numpy as np

from res.stages.PreOperation import folder_pre_operation
from res.stages.Online import online_operation, online_operation_degradation
from res.optimizer.GATools import OnGA_HyperParameters, AlphaGA_HyperParameters
import res.models.Fleet as Fleet
import res.models.Network as Network

operation_help = """Operation to perform.
(1) Pre-operation stage (deterministic)
(2) Pre-operation stage (deterministic with waiting times)
(3) Pre-operation stage (linear stochastic + deterministic CS capacities)
(4) Pre-operation stage (full linear stochastic)
(5) Online stage (open loop)
(6) Online stage (closed loop deterministic)
(7) Online stage (closed loop linear stochastic + deterministic CS capacities)
(8) Online stage (closed loop full linear stochastic)
"""
# create parser instance and mandatory arguments
parser = argparse.ArgumentParser(description='Main file. You can execute all operational stages with it.')
parser.add_argument('operation', type=int, help=operation_help)
parser.add_argument('target_folder', type=Path, help='specifies working folder')

# common options
parser.add_argument('--cs_capacities', type=int, help='Capacity of CSs. Default=2', default=2)
parser.add_argument('--repetitions', type=int, help='Experiment repetitions. Default=5', default=5)
parser.add_argument('--sat_prob_sample_time', type=int,
                    help='Sample time of CS capacity saturation probability. Default=120 (2 min)', default=120)

# pre-operation options
parser.add_argument('--additional_vehicles', type=int, help='additional EVs in pre-operation. Default=0', default=0)
parser.add_argument('--fill_up_to', type=float, help='fill the EV with a mass up to this value. Default=1.0',
                    default=1.0)

# online options
parser.add_argument('--source_folder', type=Path,
                    help='Source folder containing pre-operation results. Must be passed for simulations Default=None',
                    default=None)
parser.add_argument('--sample_time', type=float, help='Online sample time in seconds. Default=300.0', default=300.0)
parser.add_argument('--keep_times', type=int, help='How many steps keep realizations. Default=0', default=0)
parser.add_argument('--std_factor', type=float, help='Noise gain. Default: 1.0', default=1.0)
parser.add_argument('--start_earlier_by', type=float,
                    help='Start the simulation earlier by this amount of seconds. Default: 1200.0 [s] (20 min)',
                    default=1200.0)

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

ev_type = Fleet.EV.ElectricVehicle
fleet_type = Fleet.Fleet
network_type = Network.DeterministicCapacitatedNetwork
edge_type = Network.Edge.DynamicEdge

if __name__ == '__main__':
    # Pre-operation stage (deterministic)
    if args.operation == 1:
        instances_folder = args.target_folder
        if not instances_folder.is_dir():
            print("Directory is not valid: ", instances_folder)
            sys.exit(0)
        print("Will solve instances at:\n  ", instances_folder)
        input("Press any key to continue... (ctrl+Z to end process)")
        folder_pre_operation(instances_folder, args.repetitions, args.soc_policy, args.additional_vehicles,
                             args.fill_up_to, fleet_type=fleet_type, ev_type=ev_type, network_type=network_type,
                             edge_type=edge_type, cs_capacities=args.cs_capacities,
                             results_folder_suffix="deterministic", sat_prob_sample_time=args.sat_prob_sample_time)

    # Pre-operation stage (deterministic with waiting times)
    elif args.operation == 2:
        ev_type = Fleet.EV.ElectricVehicleWithWaitingTimes
        edge_type = Network.Edge.DynamicEdge

        instances_folder = args.target_folder
        if not instances_folder.is_dir():
            print("Directory is not valid: ", instances_folder)
            sys.exit(0)
        print("Will solve instances at:\n  ", instances_folder)
        input("Press any key to continue... (ctrl+Z to end process)")
        folder_pre_operation(instances_folder, args.repetitions, args.soc_policy, args.additional_vehicles,
                             args.fill_up_to, fleet_type=fleet_type, ev_type=ev_type, network_type=network_type,
                             edge_type=edge_type, cs_capacities=args.cs_capacities,
                             results_folder_suffix="deterministic_waiting_times")

    # Pre-operation stage (linear stochastic + deterministic CS capacities)
    elif args.operation == 3:
        ev_type = Fleet.EV.GaussianElectricVehicle
        edge_type = Network.Edge.GaussianEdge
        network_type = Network.DeterministicCapacitatedNetwork

        instances_folder = args.target_folder
        if not instances_folder.is_dir():
            print("Directory is not valid: ", instances_folder)
            sys.exit(0)
        print("Will solve instances at:\n  ", instances_folder)
        input("Press any key to continue... (ctrl+Z to end process)")
        folder_pre_operation(instances_folder, args.repetitions, args.soc_policy, args.additional_vehicles,
                             args.fill_up_to, fleet_type=fleet_type, ev_type=ev_type, network_type=network_type,
                             edge_type=edge_type, cs_capacities=args.cs_capacities,
                             results_folder_suffix="stochastic_deterministic")

    # Pre-operation stage (full linear stochastic)
    elif args.operation == 4:
        ev_type = Fleet.EV.GaussianElectricVehicle
        fleet_type = Fleet.GaussianFleet
        edge_type = Network.Edge.GaussianEdge
        network_type = Network.GaussianCapacitatedNetwork

        instances_folder = args.target_folder
        if not instances_folder.is_dir():
            print("Directory is not valid: ", instances_folder)
            sys.exit(0)
        print("Will solve instances at:\n  ", instances_folder)
        input("Press any key to continue... (ctrl+Z to end process)")
        folder_pre_operation(instances_folder, args.repetitions, args.soc_policy, args.additional_vehicles,
                             args.fill_up_to, fleet_type=fleet_type, ev_type=ev_type, network_type=network_type,
                             edge_type=edge_type, cs_capacities=args.cs_capacities,
                             results_folder_suffix="fully_stochastic")

    # Online stage (open loop)
    elif args.operation == 5:
        source_folder = args.source_folder
        simulations_folder = Path(args.target_folder, 'simulations_OpenLoop')
        online_operation(args.target_folder, source_folder, False, hp, args.repetitions, args.keep_times,
                         args.sample_time, args.std_factor, args.start_earlier_by, args.soc_policy, False)
################
    # Closed loop
    elif args.operation == 3:
        source_folder = Path(args.target_folder, 'source')
        simulations_folder = Path(args.target_folder, 'simulations_ClosedLoop')
        online_operation(args.target_folder, source_folder, True, hp, args.repetitions, args.keep_times,
                         args.sample_time, args.std_factor, args.start_earlier_by, args.soc_policy, False)

    # Closed loop with degradation
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
