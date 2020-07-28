from scipy.io import loadmat
import numpy as np

import res.GA_Online as ga
from res.GATools import *
import models.Network as Network
import models.OnlineFleet as Fleet
import models.Observer as obs
import time


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


def create_network_from_data(data_path: str, save_to: str) -> Network.Network:
    data = loadmat(data_path)
    network = Fleet.Network()
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


def create_fleet(save_to: str) -> Fleet.Fleet:
    ev_id = 0
    alpha_down, alpha_upp = 0, 100
    battery_capacity = 24  # kWh
    battery_capacity_nominal = battery_capacity
    max_payload = 1.2  # tonnes
    weight = 1.52  # tonnes
    max_tour_duration = 6 * 60.
    ev = Fleet.ElectricVehicle(ev_id, weight, battery_capacity, battery_capacity_nominal, alpha_upp, alpha_down,
                               max_tour_duration, max_payload)
    fleet = Fleet.Fleet({0: ev})
    fleet.write_xml(save_to, False, False, False, True)
    return fleet


class Simulator:
    def __init__(self, network_loc: str, fleet_loc: str, mat_file_loc: str, collection_file_loc: str,
                 sample_time: float, report_folder: str):
        self.network = Network.from_xml(network_loc, instance=False)
        self.fleet = Fleet.from_xml(fleet_loc, True, True, instance=False)
        self.data = loadmat(mat_file_loc)

        self.day_points = int(1440 / self.network.edges[0][0].sample_time)

        self.network_path = network_loc
        self.network_path_temp = f'{network_loc[:-4]}_temp.xml'
        self.network.write_xml(self.network_path_temp, print_pretty=True)

        self.fleet_path = fleet_loc
        self.fleet_path_temp = f'{fleet_loc[:-4]}_temp.xml'
        self.fleet.write_xml(self.fleet_path_temp, False, True, True, True, print_pretty=True)

        self.collection_path = collection_file_loc
        obs.create_collection_file(self.fleet, collection_file_loc)

        self.observer = obs.Observer(self.network_path_temp, self.fleet_path_temp, self.collection_path, report_folder,
                                     ga_time, offset_time, 2)
        self.sample_time = sample_time

    def disturb_network(self):
        for i in self.network.edges.keys():
            for j in self.network.edges.keys():
                if i == j or self.network.edges[i][j].travel_time[0] == 0.:
                    continue
                tt, ec = measure(self.data, i, j, self.day_points)
                self.network.edges[i][j].travel_time = tt
                self.network.edges[i][j].energy_consumption = ec
        self.fleet.set_network(self.network)
        self.network.write_xml(self.network_path)

    def forward_fleet(self):
        self.observer.time += self.sample_time
        for id_ev, measurement in self.observer.collection.items():
            measurement.time += self.sample_time
            measurement.time_since_start += self.sample_time
            if measurement.done:
                continue
            ev = self.fleet.vehicles[id_ev]
            (S, L) = ev.route
            if measurement.is_in_node_from:
                if measurement.time >= measurement.end_service:
                    i, j, tod = measurement.node_from, measurement.node_to, measurement.time
                    measurement.payload -= self.fleet.network.demand(i)
                    tij = self.network.t(i, j, measurement.end_service)
                    Eij = self.network.e(i, j, measurement.payload, ev.weight, measurement.end_service, tij)
                    eij = 100*Eij / ev.battery_capacity

                    measurement.is_in_node_from = False
                    measurement.eta = (measurement.time - measurement.end_service) / tij
                    measurement.soc = measurement.end_soc - measurement.eta * eij
                    measurement.consumption_since_start += Eij * measurement.eta
            else:
                i, j, tod = measurement.node_from, measurement.node_to, measurement.time
                tij = self.network.t(i, j, measurement.end_service)
                Eij = self.network.e(i, j, measurement.payload, ev.weight, measurement.end_service, tij)
                eij = 100*Eij / ev.battery_capacity
                delta = self.sample_time / tij
                if delta > 1 - measurement.eta:
                    k = S.index(j)
                    L_node_to = L[k]
                    reach_time = measurement.time + tij * (1 - measurement.eta) + ev.waiting_times1[k] - self.sample_time
                    reach_soc = measurement.soc - eij * (1 - measurement.eta)
                    spent_time = self.network.spent_time(j, reach_soc, L_node_to)

                    measurement.is_in_node_from = True
                    measurement.node_from = j
                    measurement.node_to = S[k + 1]
                    measurement.soc = reach_soc
                    measurement.end_soc = reach_soc + L_node_to
                    measurement.end_service = reach_time + spent_time
                    measurement.end_payload -= self.network.demand(j)
                    measurement.consumption_since_start += Eij * (1 - measurement.eta)
                    ev.route = (S[k:], L[k:])
                    self.fleet.write_xml(self.fleet_path_temp, False, True, True, False)
                else:
                    measurement.eta += delta
                    measurement.soc -= delta * eij
                    measurement.consumption_since_start += delta * Eij
        self.observer.write_collection()


if __name__ == '__main__':
    net_path = '../data/online/21nodes/network.xml'
    fleet_path = '../data/online/21nodes/fleet_assigned.xml'
    collection_path = '../data/online/21nodes/collection.xml'
    mat_path = '../data/online/21nodes/21_nodes.mat'
    report_folder = '../data/online/21nodes/online_no_opt/'
    ga_time = 1.
    offset_time = 2.
    sample_time = 5.
    sim = Simulator(net_path, fleet_path, mat_path, collection_path, sample_time, report_folder)

    while not sim.observer.done():
        # Observe
        n, f, current_routes = sim.observer.observe()

        if sim.observer.done():
            break
        '''
        # Optimize
        CXPB, MUTPB = 0.7, 0.9
        num_individuals = int(len(f.network)) + int(len(f) * 8) + 30
        max_generations = int(num_individuals * 1.2)
        penalization_constant = 10. * len(f.network)
        weights = (0.25, 1.0, 0.5, 1.4, 1.2)  # cost_tt, cost_ec, cost_chg_op, cost_chg_cost, cost_wait_time
        keep_best = 1  # Keep the 'keep_best' best individuals
        tournament_size = 3
        r = 3
        hp = HyperParameters(num_individuals, max_generations, CXPB, MUTPB,
                             tournament_size=tournament_size,
                             penalization_constant=penalization_constant,
                             keep_best=keep_best,
                             weights=weights,
                             r=r)
        try:
            os.mkdir(f'../data/online/21nodes/online/')
        except FileExistsError:
            pass
        save_to = f'../data/online/21nodes/online/'
        prev_best, sp = ga.code(f, hp.r)
        routes, f, bestInd, feas, tbb, data = ga.optimal_route_assignation(f, hp, sp, save_to, best_ind=prev_best,
                                                                           savefig=True)
        for ev in f.vehicles.values():
            if len(ev.route[0]) != len(current_routes[ev.id][0]):
                k = current_routes[ev.id][0].index(ev.route[0][0])
                S = current_routes[ev.id][0][:k] + ev.route[0]
                L = current_routes[ev.id][1][:k] + ev.route[1]
                ev.route = (S, L)
                ev.assigned_customers = tuple(i for i in S if f.network.isCustomer(i))

        f.write_xml(sim.fleet_path_temp, False, True, True, False)
        sim.fleet = f
        '''

        # Move vehicles and modify network
        sim.disturb_network()
        sim.forward_fleet()

