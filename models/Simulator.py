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
        ec_data = data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['soc'][0] * 24
        tt_data = data['starting_time'][0][t]['origin'][0][i]['destination'][0][j]['time'][0] / 60

        tt_average, tt_std = np.mean(tt_data), np.std(tt_data)
        ec_average, ec_std, = np.mean(ec_data), np.std(ec_data)

        tt[t] = tt_average + np.random.normal(0, tt_std)
        ec[t] = ec_average + np.random.normal(0, ec_std)

    return tt, ec


class Simulator:
    def __init__(self, network_loc: str, fleet_loc: str, mat_file_loc: str):
        self.network = net.from_xml(network_loc)
        self.fleet = fleet.from_xml(fleet_loc)
        self.data = loadmat(mat_file_loc)

        self.day_points = int(1440/self.network.edges[0][0].sample_time)

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

