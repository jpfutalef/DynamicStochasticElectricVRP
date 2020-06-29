import sys
import time
import xml.etree.ElementTree as ET

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from deap import base, creator, tools
import matplotlib.pyplot as plt

# Resources
from Fleet import from_xml
from GA_Assignation import *
from GATools import *

# %% 1. Specify instance location
#data_folder = 'data/real_data/'
data_folder = 'data/XML_files/50C_2CS_1D_3EV_3CAP/'
#instance_filename = data_folder.split('/')[-2]
instance_filename = '50C_2CS_1D_3EV_3CAP'
path = f'{data_folder}{instance_filename}.xml'

print(f'Opening:\n {path}')

# %% 2. Instance fleet
fleet = from_xml(path, assign_customers=False)
fleet.network.draw(save_to=None, width=0.02,
                   edge_color='grey', markeredgecolor='black',
                   markeredgewidth=2.0)

starting_points = {ev_id: InitialCondition(0, 0, 0, ev.alpha_up, 0) for ev_id, ev in fleet.vehicles.items()}
input('Ready! Press ENTER to continue...')

# %% 3. GA hyper-parameters
CXPB = 0.65
MUTPB = 0.85
num_individuals = 150
max_generations = 450
penalization_constant = 500000
weights = (0.2, 0.8, 1.2, 0.0)  # travel_time, charging_time, energy_consumption, charging_cost
keep_best = 1  # Keep the 'keep_best' best individuals
tournament_size = 3
allowed_charging_ops = 4

hyper_parameters = HyperParameters(num_individuals, max_generations, CXPB, MUTPB,
                                   tournament_size=tournament_size,
                                   penalization_constant=penalization_constant,
                                   keep_best=keep_best,
                                   weights=weights,
                                   allowed_charging_operations=allowed_charging_ops)
print(hyper_parameters)

# %% 4. GA toolbox
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin, feasible=False)

toolbox = base.Toolbox()

toolbox.register("individual", random_individual, num_customers=len(fleet.network.customers),
                 num_cs=len(fleet.network.charging_stations), m=len(fleet.vehicles), r=allowed_charging_ops)
toolbox.register("evaluate", fitness, fleet=fleet, starting_points=starting_points, weights=weights,
                 penalization_constant=penalization_constant, r=allowed_charging_ops)
toolbox.register("mate", crossover, m=len(fleet.vehicles), r=allowed_charging_ops, index=None)
toolbox.register("mutate", mutate, m=len(fleet.vehicles), num_customers=len(fleet.network.customers),
                 num_cs=len(fleet.network.charging_stations),
                 r=allowed_charging_ops, index=None)
toolbox.register("select", tools.selTournament, tournsize=tournament_size)
toolbox.register("select_worst", tools.selWorst)
toolbox.register("decode", decode, m=len(fleet.vehicles), fleet=fleet, starting_points=starting_points,
                 r=allowed_charging_ops)

input('Press ENTER to begin optimization...')

# %% 5. The algorithm
t_init = time.time()

# Random population
pop = [creator.Individual(toolbox.individual()) for i in range(num_individuals)]

# Evaluate the initial population and get fitness of each individual
for ind in pop:
    fit, feasible = toolbox.evaluate(ind)
    ind.fitness.values = (fit,)
    ind.feasible = feasible

print(f'  Evaluated {len(pop)} individuals')
fits = [ind.fitness.values for ind in pop]
bestOfAll = tools.selBest(pop, 1)[0]

# These will save statistics
X, Ymax, Ymin, Yavg, Ystd = [], [], [], [], []

print("################  Start of evolution  ################")
# Begin the evolution
for g in range(max_generations):
    # A new generation
    print(f"-- Generation {g}/{max_generations} --")
    X.append(g)

    # Update block probabilities
    if g < 50:
        block_probabilities = (.33, .33, .33)
    elif g < 100:
        block_probabilities = (.2, .6, .2)
    elif g < 150:
        block_probabilities = (.6, .2, .2)
    elif g < 200:
        block_probabilities = (.33, .33, .33)
    elif g < 250:
        block_probabilities = (.2, .6, .2)
    elif g < 300:
        block_probabilities = (.6, .2, .33)
    else:
        block_probabilities = (.33, .33, .33)

    # Select the best individuals, if given
    if keep_best:
        best_individuals = list(map(toolbox.clone, tools.selBest(pop, keep_best)))

    # Select and clone the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    # Mutation
    for mutant in offspring:
        if random() < MUTPB:
            toolbox.mutate(mutant, block_probability=block_probabilities)
            del mutant.fitness.values

    # Crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Evaluate the individuals with invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    for ind in invalid_ind:
        fit, feasible = toolbox.evaluate(ind)
        ind.fitness.values = (fit,)
        ind.feasible = feasible

    print(f'  Evaluated {len(invalid_ind)} individuals')

    # The population is entirely replaced by a sorted offspring
    pop[:] = offspring
    pop[:] = tools.selBest(pop, len(pop))

    # Insert best individuals from previous generation
    if keep_best:
        pop[:] = best_individuals + pop[:-keep_best]

    # Update best individual
    bestInd = tools.selBest(pop, 1)[0]
    if bestInd.fitness.wvalues[0] > bestOfAll.fitness.wvalues[0]:
        bestOfAll = bestInd

    # Real-time info
    print(f"Best individual  : {bestInd}\n Fitness: {bestInd.fitness.wvalues[0]} Feasible: {bestInd.feasible}")

    worstInd = tools.selWorst(pop, 1)[0]
    print(f"Worst individual : {worstInd}\n Fitness: {worstInd.fitness.wvalues[0]} Feasible: {worstInd.feasible}")

    print(f"Curr. best-of-all: {bestOfAll}\n Fitness: {bestOfAll.fitness.wvalues[0]} Feasible: {bestOfAll.feasible}")

    # Statistics
    fits = [sum(ind.fitness.wvalues) for ind in pop]
    mean = np.average(fits)
    std = np.std(fits)

    print(f"Max {max(fits)}")
    print(f"Min {min(fits)}")
    print(f"Avg {mean}")
    print(f"Std {std}")

    Ymax.append(-max(fits))
    Ymin.append(-min(fits))
    Yavg.append(mean)
    Ystd.append(std)

    print()

t_end = time.time()
print("################  End of (successful) evolution  ################")

algo_time = t_end - t_init
print('Algorithm time:', algo_time)


# %% Vehicles dynamics
def text_feasibility(feasible):
    if feasible:
        return 'is'
    return 'is not'


# decode best
best_fitness, best_is_feasible = toolbox.evaluate(bestOfAll)
best_routes = toolbox.decode(bestOfAll)
print(f'The best individual {text_feasibility(best_is_feasible)} feasible and its fitness is {-best_fitness}')
print('After decoding:\n', best_routes)

input('Press ENTER to continue...')
# %% Save operation if better
'''
save = False

path = folder_path + file_name + '_already_assigned.xml'

prev_op_report = res.IOTools.read_optimization_report(path)
if not prev_op_report:
    save = True

elif -best_fitness > -prev_op_report.best_fitness:
    save = True

if save:
    print('Current optimization is better... Saving!!')
    critical_points = {id_ev: (0, ev.state_leaving[0, 0], ev.state_leaving[1, 0], ev.state_leaving[2, 0]) for
                       id_ev, ev in fleet.vehicles.items()}
    fleet.save_operation_xml(path, critical_points)
    report = res.IOTools.OptimizationReport(best_fitness, best_is_feasible, t_end - t_init)
    res.IOTools.save_optimization_report(path, report)
else:
    print('Current optimization is not better.')
'''
# %% Save optimization results
import datetime
import pandas as pd

import os

now = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
folder_name = f'{now}_FEASIBLE_ASSIGNATION' if best_is_feasible else f'{now}_INFEASIBLE_ASSIGNATION/'
results_path = data_folder + folder_name

try:
    os.mkdir(results_path)
except FileExistsError:
    pass

optimization_filepath = results_path + '/optimization_info.csv'
optimization_iterations_filepath = results_path + '/optimization_iterations.csv'
theta_vector_filepath = results_path + '/nodes_occupation.csv'
info_filepath = results_path + '/hyper-parameters.txt'

# optimization info

# optimization results
df_op_gens = pd.DataFrame({'best_fitness': Ymax, 'worst_fitness': Ymin, 'pop_average': Yavg, 'pop_std': Ystd}, index=X)
df_op_gens.to_csv(optimization_iterations_filepath)

# theta vector
theta_vector = fleet.optimization_vector[fleet.optimization_vector_indices[8]:]
net_size = len(fleet.network.nodes)
events = list(range(int(len(theta_vector) / net_size)))
theta_matrix = np.array([theta_vector[i * net_size:net_size * (i + 1)] for i in events])
df_nodes_occupation = pd.DataFrame(theta_matrix, index=events)
df_nodes_occupation.to_csv(theta_vector_filepath)

# fleet operation
for id_ev, ev in fleet.vehicles.items():
    ev_filepath = results_path + f'/EV{id_ev}_operation.csv'
    reaching_data = pd.DataFrame(ev.state_reaching.T, columns=['x1_reaching', 'x2_reaching', 'x3_reaching'])
    leaving_data = pd.DataFrame(ev.state_leaving.T, columns=['x1_leaving', 'x2_leaving', 'x3_leaving'])
    route_data = pd.DataFrame({'Sk': ev.route[0], 'Lk': ev.route[1]})
    data = pd.concat([route_data, reaching_data, leaving_data], axis=1)
    data.to_csv(ev_filepath)

# costs
cost_filepath = results_path + '/costs.csv'
weight_tt, weight_ec, weight_chg_op, weight_chg_cost = weights
cost_tt, cost_ec, cost_chg_op, cost_chg_cost = fleet.cost_function()
index = ['weight', 'cost']
data = [[weight_tt, weight_ec, weight_chg_op, weight_chg_cost],
        [cost_tt, cost_ec, cost_chg_op, cost_chg_cost]]
df_costs = pd.DataFrame(data, columns=['Travel Time (min)', 'Energy Consumption (SOC)',
                                       'Charging Time (min)', 'Charging Cost'], index=index)
df_costs.to_csv(cost_filepath)

# %% save hyper-parameters
info = hyper_parameters.__str__()
info += f'Algorithm Time: {algo_time}'
info += f'\nBest individual: {bestOfAll}'
with open(info_filepath, 'w') as file:
    file.write(info)

# %% Edit assignation file
fleet.assign_customers_in_route()

assigned_path = f'{data_folder}{instance_filename}_already_assigned.xml'
tree = ET.parse(assigned_path)
_fleet = tree.find('fleet')

for _ev, ev in zip(_fleet, fleet.vehicles.values()):
    while _ev.find('assigned_customers'):
        _ev.remove(_ev.find('assigned_customers'))
    _assigned_customers = ET.SubElement(_ev, 'assigned_customers')
    for node in ev.assigned_customers:
        _node = ET.SubElement(_assigned_customers, 'node', attrib={'id': str(node)})

tree.write(assigned_path)

# %% Plot operations
plot_operation = True
if plot_operation:
    figFitness = figure(plot_width=400, plot_height=300,
                        title='Best fitness evolution')
    figFitness.circle(X, np.log(Ymax))
    figFitness.xaxis.axis_label = 'Generation'
    figFitness.yaxis.axis_label = 'log(-fitness)'

    # Standard deviation of fitness per generation
    figFitnessStd = figure(plot_width=400, plot_height=300,
                           title='Standard deviation of best fitness per generation')
    figFitnessStd.circle(X, Ystd)
    figFitnessStd.xaxis.axis_label = 'Generation'
    figFitnessStd.yaxis.axis_label = 'Standard deviation of fitness'
    figFitnessStd.left[0].formatter.use_scientific = False

    # Grid
    # p = gridplot([[figFitness, figFitnessStd]], toolbar_location='right')
    # show(p)

    fig, g = fleet.draw_operation(color_route=('r', 'b', 'g', 'c', 'y', 'r', 'b', 'g', 'c', 'y','r', 'b', 'g', 'c', 'y'),
                                  save_to=None, width=0.02,
                                  edge_color='grey', alpha=.1, markeredgecolor='black', markeredgewidth=2.0)
    fig.show()

    figs = fleet.plot_operation_pyplot()
    for k, i in enumerate(figs):
        plt.figure()
        i.tight_layout()
        i.show()
    fleet.plot_operation()


