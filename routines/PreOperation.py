from pathlib import Path
from res.stages import PreOperation
import res.models.Fleet as Fleet
import res.models.Network as Network

# %% Configuration
"""
Cases descriptions

0: full deterministic
1: stochastic with deterministic CS capacities constraints
2: full stochastic
"""

case = 0
folder_path = Path('../data/capacity_test/')
additional_vehicles = 0
fill_up_to = 1.0
soc_policy = (20, 95)
cs_capacity = 2
best_deterministic = [22, 4, 9, 16, 40, 27, 29, 12, '|', 31, 19, 44, 45, 47, 36, 32, 46, '|', 14, 6, 26, 7, 24, 21, 13, '|', 11, 41, 28, 42, 30, 25, 17, 1, 18, '|', 37, 35, 10, 3, 23, 49, 39, 48, 50, '|', 15, 38, 34, 33, 5, '|', 43, 8, 20, 2, '|', 3, 51, 11.84188595180025, -1, 51, 21.1823548565374, 13, 51, 12.04342412797305, 26, 51, 28.286610412809594, 36, 51, 6.13686003864154, 28, 51, 9.099854140476243, -1, 51, 56.22320080709538, 33, 51, 9.40698130801332, 42, 51, 14.212780543917264, 42, 51, 19.55011410855669, 10, 51, 30.080445328882767, -1, 51, 18.987129997312753, 44, 51, 25.888624721857504, 33, 51, 27.2525112805361, 38009.53708503797, 32567.96684090516, 36290.17709725042, 37182.2524306731, 39703.453741442594, 45825.69347859448, 40400.498644833504]
# best_deterministic = [43, 9, 40, 16, 7, 29, 27, 8, '|', 31, 19, 44, 47, 45, 36, 32, 46, '|', 14, 6, 26, 15, 12, 21, 13, '|', 11, 41, 28, 30, 42, 25, 17, 1, 18, '|', 37, 35, 10, 3, 23, 49, 39, 48, 50, '|', 24, 34, 38, 33, 5, '|', 4, 22, 20, 2, '|', 3, 51, 19.598628697639224, -1, 51, 25.793786257460052, 13, 51, 26.99793135756027, 26, 51, 38.169924893394196, 32, 51, 13.666212673132666, 28, 51, 15.032154492730982, -1, 51, 37.388702685312225, 33, 51, 23.03590985163261, 42, 51, 14.572729960575415, 42, 51, 20.2878460822824, 10, 51, 31.031238273213372, 16, 51, 25.424435884465538, 44, 51, 27.13251363114001, 33, 51, 28.425832817016683, 34754.237009619865, 31086.567505967265, 34381.22513506211, 38054.178530924684, 39703.453741442594, 46600.19962598529, 40780.97720857423]
sat_prob_sample_time = 120

instance_filepath = Path(folder_path, 'instance.xml')

ev_type = Fleet.EV.ElectricVehicle
fleet_type = Fleet.Fleet
network_type = Network.DeterministicCapacitatedNetwork
edge_type = Network.Edge.DynamicEdge
the_case = "deterministic"

if case == 1:
    ev_type = Fleet.EV.GaussianElectricVehicle
    edge_type = Network.Edge.GaussianEdge
    the_case = "deterministic_stochastic"

elif case == 2:
    ev_type = Fleet.EV.GaussianElectricVehicle
    fleet_type = Fleet.GaussianFleet
    network_type = Network.GaussianCapacitatedNetwork
    edge_type = Network.Edge.GaussianEdge
    the_case = "stochastic"

if __name__ == '__main__':
    PreOperation.pre_operation(None, None, instance_filepath, soc_policy, additional_vehicles, fill_up_to,
                               best_deterministic, fleet_type, ev_type, network_type, edge_type, sat_prob_sample_time,
                               cs_capacity, the_case)

