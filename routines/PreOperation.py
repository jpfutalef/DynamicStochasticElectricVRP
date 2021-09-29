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

case = 2
folder_path = Path('../data/capacity_test/')
additional_vehicles = 0
fill_up_to = 1.0
soc_policy = (20, 95)
cs_capacity = 1
best_deterministic = [43, 9, 40, 16, 7, 29, 27, 8, '|', 31, 19, 44, 47, 45, 36, 32, 46, '|', 14, 6, 26, 15, 12, 21, 13, '|', 11, 41, 28, 30, 42, 25, 17, 1, 18, '|', 37, 35, 10, 3, 23, 49, 39, 48, 50, '|', 24, 34, 38, 33, 5, '|', 4, 22, 20, 2, '|', 3, 51, 19.598628697639224, -1, 51, 25.793786257460052, 13, 51, 26.99793135756027, 26, 51, 38.169924893394196, 32, 51, 13.666212673132666, 28, 51, 15.032154492730982, -1, 51, 37.388702685312225, 33, 51, 23.03590985163261, 42, 51, 14.572729960575415, 42, 51, 20.2878460822824, 10, 51, 31.031238273213372, 16, 51, 25.424435884465538, 44, 51, 27.13251363114001, 33, 51, 28.425832817016683, 34754.237009619865, 31086.567505967265, 34381.22513506211, 38054.178530924684, 39703.453741442594, 46600.19962598529, 40780.97720857423]
sat_prob_sample_time = 10

fleet_path = Path(folder_path, 'fleet.xml')
network_path = Path(folder_path, 'network.xml')

ev_type = Fleet.EV.ElectricVehicle
fleet_type = Fleet.Fleet
network_type = Network.DeterministicCapacitatedNetwork
edge_type = Network.Edge.DynamicEdge

if case == 1:
    ev_type = Fleet.EV.GaussianElectricVehicle
    edge_type = Network.Edge.GaussianEdge

elif case == 2:
    ev_type = Fleet.EV.GaussianElectricVehicle
    fleet_type = Fleet.GaussianFleet
    network_type = Network.GaussianCapacitatedNetwork
    edge_type = Network.Edge.GaussianEdge

if __name__ == '__main__':
    PreOperation.pre_operation(fleet_path, network_path, soc_policy, additional_vehicles, fill_up_to,
                               best_deterministic, fleet_type, ev_type, network_type, edge_type, sat_prob_sample_time,
                               cs_capacity)

