from pathlib import Path
from res.stages import PreOperation

# %% Configuration
deterministic = False
folder_path = Path('../data/online/')
additional_vehicles = 0
fill_up_to = 1.0
soc_policy = (20, 95)
cs_capacity = 2
best_deterministic = [10, 4, 1, 15, 9, 22, 11, 26, 6, '|', 7, 5, 20, 12, 2, '|', 19, 3, 25, 28, 30, 21, '|', 16, 17, 29, 18, 23, 13, 24, '|', 27, 8, 14, '|', 20, 34, 18.137017818638324, 8, 33, 7.740202433050836, -1, 34, 18.90886871014706, 17, 33, 11.72194048959571, -1, 34, 28.38617587116009, 9, 32, 22.331414106697807, 23, 35, 11.002268063505833, 3, 34, 19.704560533162816, -1, 31, 2.8448931522067786, 9, 32, 29.136222759755807, 30640.382581003072, 36000.043786559654, 36445.84537065099, 42713.441074647635, 43807.592123889204]

if deterministic:
    fleet_path = Path(folder_path, 'deterministic_fleet.xml')
    network_path = Path(folder_path, 'network_det.xml')
else:
    fleet_path = Path(folder_path, 'gaussian_fleet.xml')
    network_path = Path(folder_path, 'network_sto.xml')

if __name__ == '__main__':
    PreOperation.pre_operation(fleet_path, network_path, soc_policy, additional_vehicles, fill_up_to, best_deterministic)

