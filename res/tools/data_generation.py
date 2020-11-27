import numpy as np
from scipy.io import loadmat
import res.models.Network as Network


def edge_data(data, i, j, t, average=True):
    soc_data = 1.8 * data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['soc'][0] * 24
    tt_data = 1.8 * data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['time'][0] / 60

    if average:
        soc_data = np.mean(soc_data)
        tt_data = np.mean(tt_data)

    return tt_data, soc_data


def create_network_from_data(data_path: str, save_to: str) -> Network.Network:
    data = loadmat(data_path)
    network = Network.Network()
    depot = Network.DepotNode(0)
    customers = []
    for i in range(1, 21):
        t_low = np.random.uniform(9.0 * 60, 16. * 60)
        t_upp = t_low + np.random.uniform(2. * 60, 3. * 60)
        c = Network.CustomerNode(i, np.random.uniform(8, 15),
                                 time_window_low=t_low,
                                 time_window_upp=t_upp,
                                 demand=np.random.uniform(.05, .15))
        customers.append(c)

    charging_stations = [Network.ChargeStationNode(i) for i in range(21, 22)]

    nodes = {0: depot}
    for node in customers + charging_stations:
        nodes[node.id] = node

    edges = {}
    for node_from in nodes.keys():
        edges[node_from] = {}
        for node_to in nodes.keys():
            soc_data, tt_data = [], []
            for t in range(48):
                tt, soc = edge_data(data, node_from, node_to, t, average=True)
                tt_data.append(tt if not np.isnan(tt) else 0)
                soc_data.append(soc if not np.isnan(soc) else 0)
            distance = 0.650 / tt_data[0] if tt_data[0] > 0.0 else 0.
            edges[node_from][node_to] = Network.DynamicEdge(node_from, node_to, 30, np.array(tt_data),
                                                            np.array(soc_data), distance)

    network.set_nodes(nodes)
    network.set_edges(edges)
    network.write_xml(save_to, True)
    return network
