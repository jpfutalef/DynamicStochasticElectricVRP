# %% Instance creator utility
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.io import loadmat

# rcParams['text.usetex'] = True

sys.path.append('..')
import res.models.Fleet as Fleet
import res.models.Network as Network

# %% Configuration
# Paths
main_folder = '../data/test/'
mat_path = '../data/online/instance21/init_files/21_nodes.mat'

# Network-related parameters
n_customers = 10
n_charg_sta = 4
minx, maxx = -20, 20  # km
miny, maxy = -20, 20  # km
minreq, maxreq = 0.01, 0.08
mintime, maxtime = 8, 15
mintw_low, maxtw_low = 60 * 9, 60 * 16
mintw_width, maxtw_width = 60 * 2.5, 60 * 3.5
cs_capacity = 2
fleet_size = 5
stochastic = False
print_pretty = True
hard_penalization = 0. if stochastic else 10000 * n_customers * n_charg_sta

# Factors
velocity_factor = 1.14
velocity_std_factor = 1.0

# CS technologies
tech1 = {'description': {0.0: 0.0, 75.6: 85.0, 92.4: 95.0, 122.4: 100.0},
         'technology': 0, 'technology_name': 'slow', 'price': 70}
tech2 = {'description': {0.0: 0.0, 37.2: 85.0, 46.2: 95.0, 60.6: 100.0},
         'technology': 1, 'technology_name': 'normal', 'price': 70 * 1.5}
tech3 = {'description': {0.0: 0.0, 18.6: 85.0, 23.4: 95.0, 30.6: 100.0},
         'technology': 2, 'technology_name': 'fast', 'price': 70 * 2.5}
tech_list = [tech1, tech2, tech3]

# Miscellaneous
plot_profile = False
plot_network = False
plot_charging_functions = False
fig_width, fig_height = 10, 12.941


# %% Load data
def edge_data(data, i, j, t):
    soc_data = data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['soc'][0] * 24
    tt_data = data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['time'][0] / 60
    soc_mean, tt_mean = np.mean(soc_data), np.mean(tt_data)
    soc_std, tt_std = np.std(soc_data), np.std(tt_data)
    return (soc_mean, tt_mean), (soc_std, tt_std)


data = loadmat(mat_path)

network_size = len(data['starting_time'][0][0]['origin'][0])
data_points = len(data['starting_time'][0])

soc_matrix, tt_matrix = np.zeros((1, data_points)), np.zeros((1, data_points))
samples = 0

for i in range(network_size):
    for j in range(network_size):
        if i != j:
            (soc_mean, tt_mean), (soc_std, tt_std) = edge_data(data, i, j, 0)
            dij = 35 * tt_mean / 60.  # km
            if dij == .0:
                continue
            soc_array, tt_array = np.zeros((1, data_points)), np.zeros((1, data_points))
            for t in range(data_points):
                (soc_mean, tt_mean), (soc_std, tt_std) = edge_data(data, i, j, t)
                v = 60 * dij / tt_mean  # km/h
                t1km = 60 / v  # min
                E1km = soc_mean / dij
                tt_array[0, t] = t1km
                soc_array[0, t] = E1km
            soc_matrix = np.append(soc_matrix, soc_array, axis=0)
            tt_matrix = np.append(tt_matrix, tt_array, axis=0)

# %% Obtain Values

tt_mean = np.mean(tt_matrix, axis=0)
ec_mean = np.mean(soc_matrix, axis=0)

tt_std = np.std(tt_matrix, axis=0)
ec_std = np.std(soc_matrix, axis=0)

v_mean = velocity_factor * 60 / tt_mean  # km/h
v_std = velocity_std_factor * 60 * tt_std / tt_mean ** 2

# %% Draw profile figures
if plot_profile:
    fig = plt.figure(figsize=(fig_width, fig_height / 3))
    ax1 = plt.gca()

    x = range(len(tt_mean))
    y = v_mean
    y0 = v_mean - 3 * v_std
    y1 = v_mean + 3 * v_std

    color = 'black'
    ax1.fill_between(x, y0, y1, alpha=.2, color=color, label='99.7\% confidence interval')
    ax1.plot(x, y, marker='', markersize=4, color='k', label='Average')

    ax1.set_xlabel('Time of day [x30 min]')
    ax1.set_ylabel('Velocity [km/h]')
    ax1.set_title('Velocity')
    ax1.grid(axis='y')
    ax1.legend()
    fig.tight_layout()
    fig.show()

    # fig.savefig('../data/online/instance21/1kmProfile.pdf')

if plot_charging_functions:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height / 3))
    style_list = [('Slow', '-ro'), ('Normal', '-g^'), ('Fast', '-bD')]
    for tech, (label, style) in zip(tech_list, style_list):
        t = list(tech.keys())
        soc = list(tech.values())
        ax.plot(t, soc, style, label=label)

    ax.set_xlabel('Time [min]')
    ax.set_ylabel('SOC [\%]')
    ax.set_title('Charging functions by technology')
    ax.grid(axis='y')
    ax.legend(title='Technology')
    fig.tight_layout()
    fig.show()
    # fig.savefig('../pictures/chargingFunctions.pdf')

# %% Generate Network class instance. First: nodes
nodes = {}
edges = {}
q = 0
aux = [1, 1, 0, 0]

for i in range(1 + n_customers + n_charg_sta):
    cx = float('{:.2f}'.format(np.random.uniform(aux[q - 1] * minx, aux[q - 3] * maxx)))
    cy = float('{:.2f}'.format(np.random.uniform(aux[q - 2] * miny, aux[q] * maxy)))
    q += 1
    if q == 4:
        q = 0

    # Depot
    if i == 0:
        node = Network.Node.DepotNode(0, 0, 0, 0, 0)

    # Customers
    elif 1 <= i < 1 + n_customers:
        spent_time = float('{:.2f}'.format(np.random.uniform(mintime, maxtime)))
        demand = float('{:.2f}'.format(np.random.uniform(minreq, maxreq)))
        tw_low = float('{:.2f}'.format(np.random.uniform(mintw_low, maxtw_low)))
        tw_upp = float('{:.2f}'.format(tw_low + np.random.uniform(mintw_width, maxtw_width)))
        node = Network.Node.CustomerNode(i, spent_time, demand, cx, cy, tw_low, tw_upp)

    # Charging Stations
    elif 1 <= i <= n_customers + n_charg_sta:
        j = np.random.randint(3)
        time_points, soc_points = tuple(tech_list[j].keys()), tuple(tech_list[j].values())
        node = Network.Node.ChargingStationNode(i, 0, 0, cx, cy, capacity=cs_capacity)
        node.set_technology(tech_list[j])
    # Charging station at depot
    else:
        j = np.random.randint(3)
        time_points, soc_points = tuple(tech_list[j].keys()), tuple(tech_list[j].values())
        node = Network.Node.ChargingStationNode(i, 0, 0, 0, 0, capacity=cs_capacity)
        node.set_technology(tech_list[j])

    nodes[i] = node

# Edges
for i in range(1 + n_customers + n_charg_sta):
    edges[i] = {}
    for j in range(1 + n_customers + n_charg_sta):
        a, b = nodes[i], nodes[j]
        distance = np.sqrt((a.pos_x - b.pos_x) ** 2 + (a.pos_y - b.pos_y) ** 2)
        length = distance + np.random.uniform(0, 1)
        inclination = np.deg2rad(np.random.uniform(-1, 1))
        sample_time = 30.

        length_profile = np.array([length])
        inclination_profile = np.array([inclination])

        velocity = v_mean + np.random.uniform(-.2 * min(v_mean), 0.2 * max(v_mean))

        if stochastic:
            velocity_deviation = v_std + np.random.uniform(-.2 * min(v_std), 0.2 * max(v_std))
            edge = Network.Edge.GaussianEdge(i, j, length, sample_time, velocity, length_profile, inclination_profile,
                                             'kmh/h', velocity_deviation=velocity_deviation)
        else:
            edge = Network.Edge.DynamicEdge(i, j, length, sample_time, velocity, length_profile, inclination_profile,
                                            'km/h')
        edges[i][j] = edge

# %% Instance network
if stochastic:
    network = Network.CapacitatedGaussianNetwork(nodes, edges)
else:
    network = Network.DeterministicCapacitatedNetwork(nodes, edges)

if plot_network:
    network.draw(save_to=None, width=0.003, edge_color='grey', markeredgecolor='black',
                 markeredgewidth=2.0)[0].savefig(f'{main_folder}/network.pdf')

# %% Save network
if stochastic:
    save_network_to = f'{main_folder}/network_C{n_customers}_CS{n_charg_sta}_noise{velocity_std_factor}.xml'
else:
    save_network_to = f'{main_folder}/network_C{n_customers}_CS{n_charg_sta}_deterministic.xml'
network.write_xml(save_network_to, print_pretty=print_pretty)

# %% Import network to check
network2 = Network.from_xml(save_network_to)

# %%  Create fleet
ev_id = 0
alpha_down, alpha_upp = 20, 95
battery_capacity = 24  # kWh
battery_capacity_nominal = battery_capacity
max_payload = 0.58  # tonnes
weight = 1.52  # tonnes
max_tour_duration = 7 * 60.

# %% Instance fleet
if stochastic:
    ev = Fleet.EV.GaussianElectricVehicle(ev_id, weight, battery_capacity, battery_capacity_nominal,
                                          alpha_upp, alpha_down, max_tour_duration, max_payload)
else:
    ev = Fleet.EV.ElectricVehicle(ev_id, weight, battery_capacity, battery_capacity_nominal, alpha_upp, alpha_down,
                                  max_tour_duration, max_payload)

if stochastic:
    fleet = Fleet.GaussianFleet({0: ev}, network)
else:
    fleet = Fleet.Fleet({0: ev}, network)

fleet.hard_penalization = hard_penalization
fleet.resize_fleet(fleet_size)
# %% Save fleet
fleet.write_xml(f'{main_folder}fleet.xml', network_in_file=True, assign_customers=False, with_routes=False,
                print_pretty=print_pretty)

fleet2 = Fleet.from_xml(f'{main_folder}fleet.xml')
