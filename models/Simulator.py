from scipy.io import loadmat

import models.Network as Network
import models.Observer as obs
import models.OnlineFleet as Fleet
import res.GA_Online as ga
from res.GATools import HyperParameters
from random import randint
import numpy as np
import datetime
import os

import pandas as pd


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

        tt_average, tt_std = np.mean(tt_data), 20 * np.std(tt_data)
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


def create_fleet(save_to: str) -> Fleet:
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
                 sample_time: float, create_collection: bool = True, continue_operation: bool = False):
        self.network = Network.from_xml(network_loc, instance=False)
        self.fleet = Fleet.from_xml(fleet_loc, True, True, instance=False)
        self.data = loadmat(mat_file_loc)

        self.day_points = int(1440 / self.network.edges[0][0].sample_time)

        self.network_path = network_loc
        self.network_path_temp = f'{network_loc[:-4]}_temp.xml'
        self.network.write_xml(self.network_path_temp)

        self.fleet_path = fleet_loc
        self.fleet_path_temp = f'{fleet_loc[:-4]}_temp.xml'
        if not continue_operation:
            self.fleet.write_xml(self.fleet_path_temp, False, True, True, True)

        self.collection_path = collection_file_loc
        if create_collection:
            obs.create_collection_file(self.fleet, collection_file_loc)

        self.observer = obs.Observer(self.network_path_temp, self.fleet_path_temp, self.collection_path, ga_time,
                                     offset_time, 2)
        self.sample_time = sample_time

        self.violations = {'time_window_low': [],
                           'time_window_upp': [],
                           'soc_low': [],
                           'soc_upp': []}

    def disturb_network(self):
        for i in self.network.edges.keys():
            for j in self.network.edges.keys():
                if i == j or self.network.edges[i][j].travel_time[0] == 0. or randint(0, 1):
                    continue
                tt, ec = measure(self.data, i, j, self.day_points)
                self.network.edges[i][j].travel_time = tt
                self.network.edges[i][j].energy_consumption = ec
        self.fleet.set_network(self.network)
        self.network.write_xml(self.network_path_temp)

    def forward_fleet(self):
        self.observer.time += self.sample_time
        for id_ev, measurement in self.observer.measurements.items():
            measurement.time += self.sample_time
            if measurement.done:
                continue
            i, j, tod = measurement.node_from, measurement.node_to, measurement.time
            measurement.time_since_start += self.sample_time
            ev = self.fleet.vehicles[id_ev]
            (S, L) = ev.route

            # CASE - PREVIOUS MEASUREMENT IN NODE I
            if measurement.is_in_node_from:
                # CASE - SERVICE AT NODE I ENDS AND VEHICLE STARTS MOVING AGAIN
                k = S.index(i)
                if measurement.time >= measurement.end_time + ev.waiting_times1[k]:
                    measurement.payload -= self.fleet.network.demand(i)  # Unload when stop ends
                    tij = self.network.t(i, j, measurement.end_time)
                    Eij = self.network.e(i, j, measurement.payload, ev.weight, measurement.end_time, tij)
                    eij = 100 * Eij / ev.battery_capacity

                    measurement.is_in_node_from = False
                    measurement.eta = (measurement.time - measurement.end_time) / tij
                    measurement.soc = measurement.end_soc - measurement.eta * eij
                    measurement.consumption_since_start += Eij * measurement.eta

                    if self.fleet.network.isCustomer(measurement.node_from):
                        if self.fleet.network.nodes[measurement.node_from].time_window_upp < measurement.end_time:
                            self.violations['time_window_upp'].append(1)

                    if measurement.end_soc > self.fleet.vehicles[id_ev].alpha_up:
                        self.violations['soc_upp'].append(1)

                    if measurement.end_soc < self.fleet.vehicles[id_ev].alpha_down:
                        self.violations['soc_low'].append(1)


            # CASE - MOVING ACROSS NODE I AND NODE J
            else:
                tij = self.network.t(i, j, measurement.end_time)
                Eij = self.network.e(i, j, measurement.payload, ev.weight, measurement.end_time, tij)
                eij = 100 * Eij / ev.battery_capacity
                delta = self.sample_time / tij
                # CASE - VEHICLE REACHES J
                if delta > 1 - measurement.eta:
                    measurement.is_in_node_from = True
                    measurement.node_from = j
                    k = S.index(j)

                    # CASE - J IS THE DEPOT (END TOUR)
                    if j == 0:
                        measurement.done = True
                        measurement.node_to = 0

                    else:
                        measurement.node_to = S[k + 1]

                    reach_time = measurement.time - self.sample_time + tij * (1 - measurement.eta) + ev.waiting_times0[
                        k]
                    reach_soc = measurement.soc - eij * (1 - measurement.eta)
                    spent_time = self.network.spent_time(j, reach_soc, L[k])

                    measurement.arrival_time = reach_time
                    measurement.arrival_soc = reach_soc
                    measurement.arrival_payload = measurement.end_payload

                    measurement.end_soc = reach_soc + L[k]
                    measurement.end_time = reach_time + spent_time
                    measurement.end_payload -= self.network.demand(j)

                    measurement.consumption_since_start += Eij * (1 - measurement.eta)

                    ev.route = (S[k:], L[k:])
                    self.fleet.write_xml(self.fleet_path_temp, online=True, assign_customers=True)

                    if self.fleet.network.isCustomer(j):
                        if self.fleet.network.nodes[j].time_window_low > measurement.arrival_time:
                            self.violations['time_window_low'].append(1)

                    if measurement.arrival_soc > self.fleet.vehicles[id_ev].alpha_up:
                        self.violations['soc_upp'].append(1)

                    if measurement.arrival_soc < self.fleet.vehicles[id_ev].alpha_down:
                        self.violations['soc_low'].append(1)


                # CASE - VEHICLE CONTINUES MOVING ACROSS ARC
                else:
                    measurement.eta += delta
                    measurement.soc -= delta * eij
                    measurement.consumption_since_start += delta * Eij
        now = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        # self.observer.write_collection(f'{self.report_folder}{now}.xml')
        self.observer.write_collection()


def create_folder(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


if __name__ == '__main__':
    main_folder = '../data/online/instance21/'
    mat_path = '../data/online/instance21/21_nodes.mat'
    net_path = '../data/online/instance21/network.xml'
    fleet_path = '../data/online/instance21/fleet.xml'
    num_iterations = 50

    ga_time = 1.
    offset_time = 2.
    sample_time = 5.
    soc_policy = (20, 95)

    for i in range(num_iterations):
        iteration_folder = f'{main_folder}iteration_{i + 1}/'
        no_opt_folder = f'{iteration_folder}no_opt/'
        opt_folder = f'{iteration_folder}opt/'
        collection_no_opt_path = f'{no_opt_folder}collection.xml'
        collection_opt_path = f'{opt_folder}collection.xml'

        create_folder(iteration_folder)
        create_folder(no_opt_folder)
        create_folder(opt_folder)

        # OPERATE WITHOUT OPTIMIZATION

        sim = Simulator(net_path, fleet_path, mat_path, collection_no_opt_path, sample_time)
        while not sim.observer.done():
            # Disturb network
            sim.disturb_network()

            # Forward vehicles
            sim.forward_fleet()

        nopt_df = pd.DataFrame({i: [sum(j)] for i, j in sim.violations.items()})

        # OPERATE WITH OPTIMIZATION
        sim = Simulator(net_path, fleet_path, mat_path, collection_opt_path, sample_time)
        while not sim.observer.done():
            # Disturb network
            sim.disturb_network()

            # Observe
            n, f, f_original, current_routes, ahead_routes = sim.observer.observe()

            if not ahead_routes:
                break

            # Optimize
            num_individuals = int(len(f.network) * 1.5) + int(len(f) * 10) + 50
            K1 = 100. * len(f.network) + 1000 * len(f)
            hp = HyperParameters(num_individuals=num_individuals,
                                 max_generations=num_individuals * 3,
                                 CXPB=0.7,
                                 MUTPB=0.9,
                                 weights=(0.5 / 2.218, 1. / 0.4364, 1. / 8, 1. / 80, 1.2),
                                 K1=K1,
                                 K2=K1 * 2.5,
                                 keep_best=1,
                                 tournament_size=3,
                                 r=4,
                                 alpha_up=soc_policy[1],
                                 algorithm_name='onGA',
                                 crossover_repeat=1,
                                 mutation_repeat=1)

            prev_best, critical_points = ga.code(f, ahead_routes, hp.r)
            now = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
            save_opt_to = f'{opt_folder}{now}/'
            routes, opt_data, toolbox = ga.optimal_route_assignation(f, hp, critical_points, save_opt_to,
                                                                     best_ind=prev_best, savefig=True)

            # Modify current route
            new_routes = {}
            for id_ev, ((S_old, L_old), x10, x20, x30) in current_routes.items():
                if id_ev in routes.keys():
                    S_new, L_new = routes[id_ev][0]
                    k = S_old.index(S_new[0])
                    new_routes[id_ev] = ((S_old[:k] + S_new, L_old[:k] + L_new), x10, x20, x30)

            # Send routes to vehicles
            f.set_routes_of_vehicles(new_routes, iterate=False)
            f_original.update_from_another_fleet(f)
            f_original.write_xml(sim.observer.fleet_path, online=True, assign_customers=True)

            # Forward vehicles
            sim.forward_fleet()

        opt_df = pd.DataFrame({i: [sum(j)] for i, j in sim.violations.items()})

        # SAVE RESULTS
        with pd.ExcelWriter(f'{iteration_folder}violations.xlsx') as writer:
            nopt_df.to_excel(writer, sheet_name='NoOpt')
            opt_df.to_excel(writer, sheet_name='Opt')
