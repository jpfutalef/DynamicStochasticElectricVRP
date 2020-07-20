from scipy.io import loadmat
import numpy as np

import models.Network as net
import models.Fleet as fleet


def measure(data, i, j, day_points):
    """
    Reads data object, and returns energy consumption and travel times across arc ij with gaussian noise applied.
    :param data: mat file handler
    :param i: node from id
    :param j: node to id
    :param day_points: number of measurements per day
    :return: measurements (travel time, energy consumption) in minutes and kWh
    """
    tt, ec = np.zeros(day_points), np.zeros(day_points)
    for t in range(day_points):
        ec_data = 1.8 * data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['soc'][0] * 24
        tt_data = 1.8 * data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['time'][0] / 60

        tt_average, tt_std = np.mean(tt_data), np.std(tt_data)
        ec_average, ec_std, = np.mean(ec_data), np.std(ec_data)

        tt[t] = tt_average + np.random.normal(0, tt_std)
        ec[t] = ec_average + np.random.normal(0, ec_std)

    return tt, ec


def edge_data(data, i, j, t, average=True):
    soc_data = 1.8 * data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['soc'][0] * 24
    tt_data = 1.8 * data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['time'][0] / 60

    if average:
        soc_data = np.mean(soc_data)
        tt_data = np.mean(tt_data)

    return tt_data, soc_data


def create_network_from_data(data_path: str, save_to: str):
    data = loadmat(data_path)
    network = fleet.Network()
    depot = net.DepotNode(0)
    customers = []
    for i in range(1, 21):
        t_low = np.random.uniform(9.0 * 60, 16. * 60)
        t_upp = t_low + np.random.uniform(2. * 60, 3. * 60)
        c = net.CustomerNode(i, np.random.uniform(8, 15),
                             time_window_low=t_low,
                             time_window_upp=t_upp,
                             demand=np.random.uniform(.05, .15))
        customers.append(c)

    charging_stations = [net.ChargeStationNode(i) for i in range(21, 22)]

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
            edges[node_from][node_to] = net.DynamicEdge(node_from, node_to, 30, np.array(tt_data), np.array(soc_data),
                                                        distance)

    network.set_nodes(nodes)
    network.set_edges(edges)
    network.write_xml(save_to, True)


class Simulator:
    def __init__(self, network_loc: str, fleet_loc: str, mat_file_loc: str):
        self.network = net.from_xml(network_loc)
        self.fleet = fleet.from_xml(fleet_loc)
        self.data = loadmat(mat_file_loc)

        self.day_points = int(1440 / self.network.edges[0][0].sample_time)

        self.network_path = network_loc
        self.network_path_temp = f'{network_loc[-4]}_temp.xml'
        self.network.write_xml(self.network_path_temp)

        self.fleet_path = fleet_loc
        self.fleet_path_temp = f'´{fleet_loc[-4]}_temp.xml'
        self.fleet.write_xml(self.fleet_path_temp, False, True, True)

    def disturb_network(self):
        for i in self.network.edges.keys():
            for j in self.network.edges.keys():
                if i == j or self.network.edges[i][j].travel_time[0] == 0.:
                    continue
                tt, ec = measure(self.data, i, j, self.day_points)
                self.network.edges[i][j].travel_time = tt
                self.network.edges[i][j].energy_consumption = ec

        self.network.write_xml(self.network_path)

    def forward_fleet(self):
        pass

if __name__ == '__main__':
    create_network_from_data('/')