from pathlib import Path
from res.stages import PreOperation

# %% Configuration
deterministic = False
folder_path = Path('../data/test/')
network_path = Path(folder_path, 'network.xml')
additional_vehicles = 0
fill_up_to = 0.95
soc_policy = (20, 95)
best_deterministic = [15, 11, 9, 18, 6, 19, 8, 16, 12, '|', 20, 5, 7, 13, 17, 3, '|', 4, 1, 14, 2, 10, '|',
                      -1, 21, 27.592116389112086, -1, 21, 39.64069162665806, 11, 22, 28.093034096484203,
                      -1, 22, 31.47735281603975, -1, 21, 20.11477727993111, -1, 22, 29.500787897206905,
                      6, 22, 29.21742510833294, 20, 22, 15.95248391530103, -1, 21, 15.855166986651374,
                      34853.44802078326, 40747.45622712875, 49676.57869889013]

if deterministic:
    fleet_path = Path(folder_path, 'deterministic_fleet.xml')
else:
    fleet_path = Path(folder_path, 'gaussian_fleet.xml')

if __name__ == '__main__':
    PreOperation.pre_operation(fleet_path, network_path, soc_policy, additional_vehicles, fill_up_to, best_deterministic)

