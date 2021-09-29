from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans

import res.models.Edge as Edge
import res.models.ElectricVehicle as EV
import res.models.Fleet as Fleet
import res.models.Network as Network
import res.models.Node as Node

"""
PARAMETERS FOR THE DEVELOPMENT OF INSTANCES
"""
instance_path = Path('data/instances/c100_cs10_15x15km.xml')
mat_path = Path('data/mat_files/Santiago22.mat')

fig_width, fig_height = 10, 12.941
full_figratio = (fig_width, fig_height)

# Network parameters
n_customers = 100
n_charg_sta = 10
# minx, maxx = -20000, 20000 # m
# miny, maxy = -20000, 20000 # m
min_radius, max_radius = 500, 15000  # m
min_theta, max_theta = 0, 2 * np.pi
minreq, maxreq = 10, 50  # kg
mintime, maxtime = 60 *5, 60 * 15  # s
mintw_low, maxtw_low = 8 * 60 * 60, 13 * 60 * 60  # s
mintw_width, maxtw_width = 3 * 60 * 60, 5 * 60 * 60  # s
cs_capacity = 2
std_factor = 1
show_velocity_realizations = False


tech1 = {0.0: 0.0, 75.6 * 60: 85.0, 92.4 * 60: 95.0, 122.4 * 60: 100.0}  # slow
tech2 = {0.0: 0.0, 37.2 * 60: 85.0, 46.2 * 60: 95.0, 60.6 * 60: 100.0}  # normal
tech3 = {0.0: 0.0, 18.6 * 60: 85.0, 23.4 * 60: 95.0, 30.6 * 60: 100.0}  # fast
tech_list = [tech1, tech2, tech3]
tech_price = [70., 70. * 1.5, 70. * 2.5]
tech_name = ["Slow", "Normal", "Fast"]
which_technology = None # If None, choose random between tech_list
show_charging_function = False
cs_at_depot = True

# Fleet parameters
alpha_down, alpha_upp = 20, 95
battery_capacity = 24000 * 3600  # J
battery_capacity_nominal = battery_capacity
max_payload = 553  # kg
ev_weight = 1.52 * 1000  # kg
max_tour_duration = 6 * 60. * 60  # s

"""
USEFUL FUNCTIONS AND VARIABLES
"""


def edge_data(data, i, j, t):
    soc_data = data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['soc'][0] * 24 / (1000 * 3600)  # Ws
    tt_data = data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['time'][0]  # s
    soc_mean, tt_mean = np.mean(soc_data), np.mean(tt_data)
    soc_std, tt_std = np.std(soc_data), np.std(tt_data)
    return (soc_mean, tt_mean), (soc_std, tt_std)

cs_at_depot_off = -1 if cs_at_depot else 0

"""
CREATION OF NETWORK
"""
data = loadmat(mat_path)

network_size = len(data['starting_time'][0][0]['origin'][0])
data_points = len(data['starting_time'][0])

v_mean_matrix = np.zeros((1, data_points))
v_std_matrix = np.zeros((1, data_points))

for i in range(network_size):
    for j in [x for x in range(network_size) if x != i]:
        (soc_mean, tt_mean), (soc_std, tt_std) = edge_data(data, i, j, 0)
        v_avg = 40 * 1000 / 3600
        dij = tt_mean * v_avg  # m
        if dij == .0:
            continue
        soc_array_mean, tt_array_mean, v_array_mean = np.zeros((1, data_points)), np.zeros(
            (1, data_points)), np.zeros((1, data_points))
        soc_array_std, tt_array_std, v_array_std = np.zeros((1, data_points)), np.zeros((1, data_points)), np.zeros(
            (1, data_points))
        for t in range(data_points):
            (soc_mean, tt_mean), (soc_std, tt_std) = edge_data(data, i, j, t)
            t1km_mean = tt_mean / dij  # s
            t1km_std = tt_std / dij  # s

            v_array_mean[0, t] = dij / tt_mean
            v_array_std[0, t] = 7 * std_factor * tt_std * (dij / tt_mean) ** 2 / dij

        v_mean_matrix = np.append(v_mean_matrix, v_array_mean, axis=0)
        v_std_matrix = np.append(v_std_matrix, v_array_std, axis=0)

v_mean_matrix = v_mean_matrix[1:, :]
v_std_matrix = v_std_matrix[1:, :]


# Show velocity statistic realizations
if show_velocity_realizations:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height / 3))
    for i in range(v_mean_matrix.shape[0]):
        # v_index = np.random.randint(0, v_mean_matrix.shape[0])
        v_index = i
        ax1.plot(v_mean_matrix[v_index, :] * 3.6)
        ax1.set_xlabel("Sample instant")
        ax1.set_ylabel("Velocity average [km/h]")

        ax2.plot(v_std_matrix[v_index, :] * 3.6)
        ax2.set_xlabel("Sample instant")
        ax2.set_ylabel("Velocity std [km/h]")
    fig.show()


# Plot charging functions
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

if show_charging_function:
    fig.show()
else:
    plt.close(fig)

# Dictionaries with nodes and edges
nodes = {}
edges = {}

customer_locations = []

for i in range(1 + n_customers):
    r = np.random.uniform(min_radius, max_radius)
    theta = np.random.uniform(min_theta, max_theta)
    # cx = float('{:.2f}'.format(np.random.uniform(minx, maxx)))
    # cy = float('{:.2f}'.format(np.random.uniform(miny, maxy)))
    cx = float('{:.2f}'.format(r * np.cos(theta)))
    cy = float('{:.2f}'.format(r * np.sin(theta)))

    # Depot
    if i == 0:
        node = Node.DepotNode(0, 0, 0, 0, 0)

    # Customers
    else:
        spent_time = float('{:.2f}'.format(np.random.uniform(mintime, maxtime)))
        demand = float('{:.2f}'.format(np.random.uniform(minreq, maxreq)))
        tw_low = float('{:.2f}'.format(np.random.uniform(mintw_low, maxtw_low)))
        tw_upp = float('{:.2f}'.format(tw_low + np.random.uniform(mintw_width, maxtw_width)))
        node = Node.CustomerNode(i, spent_time, demand, cx, cy, tw_low, tw_upp)

        customer_locations.append([cx, cy])

    # Charging Stations
        """
        elif 1 <= i < n_customers + n_charg_sta + 1 + cs_at_depot_off:
            j = np.random.randint(3) if which_technology is None else which_technology
            time_points, soc_points = tuple(tech_list[j].keys()), tuple(tech_list[j].values())
            node = Node.ChargingStationNode(i, pos_x=cx, pos_y=cy, capacity=cs_capacity, time_points=time_points,
                                            soc_points=soc_points, technology=j + 1, technology_name=tech_name[j],
                                            price=tech_price[j])
        """
    nodes[i] = node

# Recharging stations using clusters
n_clusters = n_charg_sta - 1 if cs_at_depot else n_charg_sta

if n_clusters:
    X = np.array(customer_locations)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    for i_, (cx, cy) in enumerate(kmeans.cluster_centers_):
        i = i_ + n_customers + 1
        j = np.random.randint(3) if which_technology is None else which_technology
        time_points, soc_points = tuple(tech_list[j].keys()), tuple(tech_list[j].values())
        node = Node.ChargingStationNode(i, pos_x=cx, pos_y=cy, capacity=cs_capacity, time_points=time_points,
                                        soc_points=soc_points, technology=j + 1, technology_name=tech_name[j],
                                        price=tech_price[j])
        nodes[i] = node

if cs_at_depot:
    i = n_customers + n_charg_sta
    j = np.random.randint(3) if which_technology is None else which_technology
    time_points, soc_points = tuple(tech_list[j].keys()), tuple(tech_list[j].values())
    node = Node.ChargingStationNode(i, pos_x=0., pos_y=0., capacity=cs_capacity, time_points=time_points,
                                    soc_points=soc_points, technology=j + 1, technology_name=tech_name[j],
                                    price=tech_price[j])
    nodes[i] = node

# Edges
for i in range(1 + n_customers + n_charg_sta):
    edges[i] = {}
    for j in range(1 + n_customers + n_charg_sta):
        a, b = nodes[i], nodes[j]
        dist = np.sqrt((a.pos_x - b.pos_x) ** 2 + (a.pos_y - b.pos_y) ** 2)
        if dist:
            v_index = np.random.randint(0, v_mean_matrix.shape[0])
            v_mean = v_mean_matrix[v_index, :]
            v_std = v_std_matrix[v_index, :]
        else:
            v_mean = np.zeros_like(v_mean_matrix[0, :])
            v_std = np.zeros_like(v_mean_matrix[0, :])
        length_profile = np.array([dist])
        inclination_profile = np.array([0])

        edge = Edge.GaussianEdge(i, j, dist, int(24*60*60/len(v_mean)), v_mean, length_profile, inclination_profile,
                                 velocity_deviation=v_std)

        edges[i][j] = edge

# Instance network
network = Network.GaussianCapacitatedNetwork(nodes, edges)
network.draw(save_to=None, width=0.003, edge_color='grey')
plt.show()


"""
CREATION OF FLEET WITH ONE EV
"""
ev = EV.GaussianElectricVehicle(0, ev_weight, battery_capacity, battery_capacity_nominal, alpha_upp, alpha_down,
                                max_tour_duration, max_payload)

fleet = Fleet.GaussianFleet({0: ev})

"""
CREATION OF INSTANCE
"""
fleet.set_network(network)
fleet.write_xml(instance_path, True)
