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
from GA_AlreadyAssigned import *
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

# %%
# 1. Specify file
path = './data/GA_implementation_xml/20C_4CS_1D_4EV/20C_4CS_1D_4EV_already_assigned.xml'
print('Opening:', path)

# %% 3. Instance fleet
init_soc = 80.
init_node = 0
all_charging_ops = 2

fleet = from_xml(path, assign_customers=True)

customers_to_visit = {ev_id: ev.assigned_customers for ev_id, ev in fleet.vehicles.items()}

starting_points = {ev_id: InitialCondition(init_node, 0, 0, init_soc, sum([fleet.network.nodes[x].demand
                                                                           for x in ev.assigned_customers]))
                   for ev_id, ev in fleet.vehicles.items()}

input('Press enter to continue...')

# %%
# 7. GA hyperparameters
CXPB, MUTPB = 0.6, 0.8
n_individuals = 50
generations = 50
penalization_constant = 500000
weights = (1.0, 1.1, 0.0, 0.0)  # travel_time, charging_time, energy_consumption, charging_cost
keep_best = 1  # Keep the 'keep_best' best individuals

# Arguments
indices = block_indices(customers_to_visit, allowed_charging_operations=all_charging_ops)
common_args = {'allowed_charging_operations': all_charging_ops, 'indices': indices}
# %%
# Fitness objects
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin, feasible=False)

# Toolbox
toolbox = base.Toolbox()

toolbox.register("individual", random_individual,
                 starting_points=starting_points, customers_to_visit=customers_to_visit,
                 charging_stations=fleet.network.charging_stations, **common_args)
toolbox.register("evaluate", fitness,
                 fleet=fleet, starting_points=starting_points, weights=weights,
                 penalization_constant=penalization_constant, **common_args)
toolbox.register("mate", crossover, index=None, **common_args)
toolbox.register("mutate", mutate,
                 starting_points=starting_points, customers_to_visit=customers_to_visit,
                 charging_stations=fleet.network.charging_stations, index=None, **common_args)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("select_worst", tools.selWorst)
toolbox.register("decode", decode,
                 starting_points=starting_points, **common_args)

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
    print("-- Generation %i --" % g)

    # Select the best individuals, if given
    if keep_best:
        best_individuals = list(map(toolbox.clone, tools.selBest(pop, keep_best)))

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))

    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        # cross two individuals with probability CXPB
        if random.random() < CXPB:
            toolbox.mate(child1, child2)

            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        # mutate an individual with probability MUTPB
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
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

# %% Save operation if better
plot_operation = False

save = False

prev_op_report = res.IOTools.read_optimization_report(path)
if not prev_op_report:
    save = True

elif -best_fitness > -prev_op_report.best_fitness:
    save = True

if save:
    critical_points = {id_ev: (0, ev.state_leaving[0, 0], ev.state_leaving[1, 0], ev.state_leaving[2, 0]) for
                       id_ev, ev in fleet.vehicles.items()}
    fleet.save_operation_xml(path, critical_points)
    report = res.IOTools.OptimizationReport(best_fitness, best_is_feasible, t_end-t_init)
    res.IOTools.save_optimization_report(path, report)

if not plot_operation:
    import sys
    sys.exit(1)

# %% Fitness per generation
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
fleet.plot_operation()
