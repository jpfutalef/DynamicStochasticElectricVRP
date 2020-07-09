# %%

# Too work with arguments and script paths
import sys

# scientific libraries and utilities
import numpy as np
import random
import time
import copy

# GA library
from deap import base
from deap import creator
from deap import tools

# Resources
from GA_AlreadyAssigned3 import *
from Fleet import from_xml, InitialCondition
import res.IOTools

# Visualization tools
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.models.annotations import Arrow, Label
from bokeh.models.arrow_heads import VeeHead
from bokeh.models import Whisker, Span, Range1d

t0 = time.time()

sys.path.append('..')

# %% 1. Specify instance location
# data_folder = 'data/real_data/'
data_folder = 'data/XML_files/35C_2CS_1D_3EV_4CAP_HIGHWEIGHT/08-07-2020_13-25-11_FEASIBLE_ASSIGNATION/'
# instance_filename = data_folder.split('/')[-2]
instance_filename = 'assigned'
path = f'{data_folder}{instance_filename}.xml'

print(f'Opening:\n {path}')

# %% 2. Instance fleet
fleet = from_xml(path, assign_customers=True)
fleet.network.draw(save_to=None, width=0.02,
                   edge_color='grey', markeredgecolor='black',
                   markeredgewidth=2.0)
input('Press enter to continue...')

# %% 3. GA hyperparameters
all_charging_ops = 2
customers_to_visit = {ev_id: ev.assigned_customers for ev_id, ev in fleet.vehicles.items()}

starting_points = {ev_id: InitialCondition(0, 0, 0, ev.alpha_up, sum([fleet.network.nodes[x].demand
                                                                           for x in ev.assigned_customers]))
                   for ev_id, ev in fleet.vehicles.items()}
CXPB, MUTPB = 0.65, 0.85
n_individuals = 90
generations = 150
penalization_constant = 500000
weights = (0.2, 0.8, 1.2, 0.0)  # travel_time, charging_time, energy_consumption, charging_cost
keep_best = 1  # Keep the 'keep_best' best individuals
#tournament_size = int(n_individuals * 0.1)
tournament_size = 3
block_probability = (.6, .2, .2)
repeat_crossover = 1
repeat_mutation = 1

info = f'''
Hyper-parameters:
CXPB, MUTPB = {CXPB}, f{MUTPB}
n_individuals = {n_individuals}
generations = {generations}
penalization_constant = {penalization_constant}
weights = {weights}
keep_best = {keep_best}
tournament_size = {tournament_size}
block_probability = {block_probability}]
repeat_crossover = {repeat_crossover}
repeat_mutation = {repeat_mutation}
'''

# Arguments
indices = block_indices(customers_to_visit, all_charging_ops)
# %%
# Fitness objects
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin, feasible=False)

# Toolbox
toolbox = base.Toolbox()

toolbox.register("individual", random_individual, customers_per_vehicle=customers_to_visit,
                 charging_stations=fleet.network.charging_stations, r=all_charging_ops)
toolbox.register("evaluate", fitness, fleet=fleet, indices=indices, init_state=starting_points, r=all_charging_ops, weights=weights,
                 penalization_constant=penalization_constant)
toolbox.register("mate", crossover, indices=indices, repeat=repeat_crossover)
toolbox.register("mutate", mutate, indices=indices, charging_stations=fleet.network.charging_stations,
                 repeat=repeat_mutation)
toolbox.register("select", tools.selTournament, tournsize=tournament_size)
toolbox.register("select_worst", tools.selWorst)
toolbox.register("decode", decode, indices=indices, init_state=starting_points, r=all_charging_ops)

# %% the algorithm
tInitGA = time.time()
# Population TODO create function

pop = []
for i in range(0, n_individuals):
    pop.append(creator.Individual(toolbox.individual()))

# Evaluate the entire population
# fitnesses = list(map(toolbox.evaluate, pop))

for ind in pop:
    fit, feasible = toolbox.evaluate(ind)
    ind.fitness.values = (fit,)
    ind.feasible = feasible

print("  Evaluated %i individuals" % len(pop))

# Extracting all the fitnesses of
fits = [ind.fitness.values for ind in pop]

# Variable keeping track of the number of generations
g = 0
Ymax = []
Ymin = []
Yavg = []
Ystd = []
X = []

bestOfAll = tools.selBest(pop, 1)[0]

print("################  Start of evolution  ################")
t_init = time.time()
# Begin the evolution
while g < generations:
    # A new generation
    g = g + 1
    X.append(g)
    print(f"-- Generation {g}/{generations} --")

    # Select the best individuals, if given
    if keep_best:
        best_individuals = list(map(toolbox.clone, tools.selBest(pop, keep_best)))

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))

    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # blocks probabilities
    if g < 90:
        block_probability = (.33, .33, .33)
    elif g < 150:
        block_probability = (.2, .6, .2)
    elif g < 210:
        block_probability = (.6, .2, .2)
    else:
        block_probability = (.33, .33, .33)

    # Crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        # cross two individuals with probability CXPB
        if random() < CXPB:
            toolbox.mate(child1, child2, block_probability=block_probability)

            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values

    # Mutation
    for mutant in offspring:
        # mutate an individual with probability MUTPB
        if random() < MUTPB:
            toolbox.mutate(mutant, block_probability=block_probability)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    for ind in invalid_ind:
        fit, feasible = toolbox.evaluate(ind)
        ind.fitness.values = (fit,)
        ind.feasible = feasible

    print("  Evaluated %i individuals" % len(invalid_ind))

    # The population is entirely replaced by a sorted offspring
    pop[:] = offspring
    pop[:] = tools.selBest(pop, len(pop))

    # Remove worst individuals, if keeping the best of them
    if keep_best:
        pop[keep_best:] = pop[:-keep_best]
        pop[:keep_best] = best_individuals

    # Show worst and best
    bestInd = tools.selBest(pop, 1)[0]
    print(f"Best individual: {bestInd} \n        Fitness: {bestInd.fitness.wvalues[0]}  Feasible: {bestInd.feasible}")

    worstInd = tools.selWorst(pop, 1)[0]
    print(
        f"Worst individual: {worstInd} \n        Fitness: {worstInd.fitness.wvalues[0]}  Feasible: {worstInd.feasible}")

    print(
        f"Best of all individuals: {bestOfAll} \n        Fitness: {bestOfAll.fitness.wvalues[0]}  Feasible: {bestOfAll.feasible}")

    # Statistics
    fits = [sum(ind.fitness.wvalues) for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print("  Max %s" % max(fits))
    print("  Min %s" % min(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)

    Ymax.append(-max(fits))
    Ymin.append(-min(fits))
    Yavg.append(mean)
    Ystd.append(std)

    # Save best ind
    if bestInd.fitness.wvalues[0] > bestOfAll.fitness.wvalues[0]:
        bestOfAll = bestInd
# %%
t_end = time.time()
print("################  End of (successful) evolution  ################")

algo_time = t_end-t_init
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
print(f'After decoding:\n{best_routes}')

# %% Save operation if better
save = True

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
    report = res.IOTools.OptimizationReport(best_fitness, best_is_feasible, t_end-t_init)
    res.IOTools.save_optimization_report(path, report)
else:
    print('Current optimization is not better.')

# %% Save optimization results
import datetime
import pandas as pd
import os

now = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
folder_name = f'{now}_FEASIBLE_ASSIGNED' if best_is_feasible else f'{now}_INFEASIBLE_ASSIGNED/'
results_path = folder_path + folder_name

try:
    os.mkdir(results_path)
except FileExistsError:
    pass

optimization_filepath = results_path + '/optimization_info.csv'
optimization_iterations_filepath = results_path + '/optimization_iterations.csv'
theta_vector_filepath = results_path + '/nodes_occupation.csv'
cost_filepath = results_path + '/costs.csv'
info_filepath = results_path + '/hyper-parameters.txt'

# optimization info

# optimization results
df_op_gens = pd.DataFrame({'best_fitness': Ymax, 'worst_fitness': Ymin, 'pop_average': Yavg, 'pop_std': Ystd}, index=X)
df_op_gens.to_csv(optimization_iterations_filepath)

# theta vector
theta_vector = fleet.optimization_vector[fleet.optimization_vector_indices[8]:]
net_size = len(fleet.network.nodes)
events = list(range(int(len(theta_vector)/net_size)))
theta_matrix = np.array([theta_vector[i*net_size:net_size*(i+1)] for i in events])
df_nodes_occupation = pd.DataFrame(theta_matrix, index=events)
df_nodes_occupation.to_csv(theta_vector_filepath)

# costs
weight_tt, weight_ec, weight_chg_op, weight_chg_cost = weights
cost_tt, cost_ec, cost_chg_op, cost_chg_cost = fleet.cost_function()
index = ['weight', 'cost']
data = [[weight_tt, weight_ec, weight_chg_op, weight_chg_cost],
        [cost_tt, cost_ec, cost_chg_op, cost_chg_cost]]
df_costs = pd.DataFrame(data, columns=['Travel Time (min)', 'Energy Consumption (SOC)',
                                       'Charging Time (min)', 'Charging Cost'], index=index)
df_costs.to_csv(cost_filepath)

# fleet operation
for id_ev, ev in fleet.vehicles.items():
    ev_filepath = results_path + f'/EV{id_ev}_operation.csv'
    reaching_data = pd.DataFrame(ev.state_reaching.T, columns=['x1_reaching', 'x2_reaching', 'x3_reaching'])
    leaving_data = pd.DataFrame(ev.state_leaving.T, columns=['x1_leaving', 'x2_leaving', 'x3_leaving'])
    route_data = pd.DataFrame({'Sk': ev.route[0], 'Lk': ev.route[1]})
    data = pd.concat([route_data, reaching_data, leaving_data], axis=1)
    data.to_csv(ev_filepath)


# save hyper-parameters
info += f'\nAlgorithm Time: {algo_time}'
info += f'\nBest individual: {bestOfAll}'
with open(info_filepath, 'w') as file:
    file.write(info)
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
    p = gridplot([[figFitness, figFitnessStd]], toolbar_location='right')
    show(p)

    # %%
    #fleet.plot_operation()


