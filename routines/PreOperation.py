from pathlib import Path
from res.stages import PreOperation

# %% Configuration
deterministic = False
folder_path = Path('../data/capacity_test/')
additional_vehicles = 0
fill_up_to = 1.0
soc_policy = (20, 95)
cs_capacity = 2
best_deterministic = [43, 9, 40, 16, 7, 29, 27, 8, '|', 31, 19, 44, 47, 45, 36, 32, 46, '|', 14, 6, 26, 15, 12, 21, 13, '|', 11, 41, 28, 30, 42, 25, 17, 1, 18, '|', 37, 35, 10, 3, 23, 49, 39, 48, 50, '|', 24, 34, 38, 33, 5, '|', 4, 22, 20, 2, '|', 3, 51, 19.598628697639224, -1, 51, 25.793786257460052, 13, 51, 26.99793135756027, 26, 51, 38.169924893394196, 32, 51, 13.666212673132666, 28, 51, 15.032154492730982, -1, 51, 37.388702685312225, 33, 51, 23.03590985163261, 42, 51, 14.572729960575415, 42, 51, 20.2878460822824, 10, 51, 31.031238273213372, 16, 51, 25.424435884465538, 44, 51, 27.13251363114001, 33, 51, 28.425832817016683, 34754.237009619865, 31086.567505967265, 34381.22513506211, 38054.178530924684, 39703.453741442594, 46600.19962598529, 40780.97720857423]

if deterministic:
    fleet_path = Path(folder_path, 'deterministic_fleet.xml')
    network_path = Path(folder_path, 'network_det.xml')
else:
    fleet_path = Path(folder_path, 'gaussian_fleet.xml')
    network_path = Path(folder_path, 'network_sto.xml')

if __name__ == '__main__':
    PreOperation.pre_operation(fleet_path, network_path, soc_policy, additional_vehicles, fill_up_to, best_deterministic)

