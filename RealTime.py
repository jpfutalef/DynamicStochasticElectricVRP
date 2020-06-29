# %% Imports
# scientific libraries and utilities
import random
import sys
import time

# Visualization tools
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show

# GA library
from deap import base
from deap import creator
from deap import tools

# Resources
from GA_RealTime import *
from res.GATools import *
import res.IOTools
from Fleet import from_xml

t0 = time.time()
sys.path.append('..')

# %%
# 1. Specify file
path = './data/GA_implementation_xml/10C_2CS_1D_2EV/10C_2CS_1D_2EV_realtime.xml'
print('Opening:', path)

# %% 2. Instance fleet
all_charging_ops = 2
fleet = from_xml(path, assign_customers=True)
input('Press enter to continue...')

# %% 3. GA hyper parameters
CXPB, MUTPB = 0.6, 0.85
n_individuals = 80
generations = 150
penalization_constant = 500000
weights = (1.0, 1.1, 0.0, 0.0)  # travel_time, charging_time, energy_consumption, charging_cost
keep_best = 1 # Keep the 'keep_best' best individuals
warm_start = True
warm_start_size = 3

print('  Summary of GA hyper-parameters:')
print(f"""CXP  =  {CXPB}
MUTPB  =  {MUTPB}
Population size  =  {n_individuals}
Max. generations  =  {generations}
Penalization constants  =  {penalization_constant}
Weights  =  {weights}
Keeping the best(s) {keep_best} individual(s)
Warm start: {warm_start}
Warm start with {warm_start_size} individuals and the best from previous optimization""")

# %% 4. Realtime Loop
# Fitness objects
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin, feasible=False)

while True:
    # %% Observe
    input('Update XML file and press ENTER...')
    fleet.update_from_xml(path, do_network=True)

    # %% Check if there are vehicles to route
    if not len(fleet.vehicles):
        break

    # %% Create toolbox
    customers_to_visit = {ev_id: ev.assigned_customers for ev_id, ev in fleet.vehicles.items()}
    indices = block_indices(customers_to_visit, allowed_charging_operations=all_charging_ops)
    common_args = {'allowed_charging_operations': all_charging_ops, 'indices': indices}

    toolbox = base.Toolbox()
    toolbox.register("individual", random_individual,
                     starting_points=fleet.starting_points, customers_to_visit=customers_to_visit,
                     charging_stations=fleet.network.charging_stations, **common_args)
    toolbox.register("evaluate", fitness,
                     fleet=fleet, starting_points=fleet.starting_points, weights=weights,
                     penalization_constant=penalization_constant, **common_args)
    toolbox.register("mate", crossover, index=None, **common_args)
    toolbox.register("mutate", mutate,
                     starting_points=fleet.starting_points, customers_to_visit=customers_to_visit,
                     charging_stations=fleet.network.charging_stations, index=None, **common_args)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("select_worst", tools.selWorst)
    toolbox.register("decode", decode,
                     starting_points=fleet.starting_points, **common_args)
    toolbox.register("best_prev", code, fleet, all_charging_ops)
    input('Updated! Press ENTER to begin OPTIMIZATION...')

    # %% Setup algorithm
    t_init = time.time()

    # Get the best individual from previous optimization
    best_prev = creator.Individual(toolbox.best_prev())
    best_fitness_prev, feasible_prev = toolbox.evaluate(best_prev)

    # Warm start with mutated individuals from best
    warm_start_individuals = []
    for x in range(warm_start_size):
        new_warm_ind = toolbox.mutate(toolbox.clone(best_prev))
        warm_start_individuals.append(new_warm_ind)

    if warm_start:
        # Create random population
        pop = [creator.Individual(toolbox.individual()) for i in range(n_individuals - warm_start_size - 1)]

        # Add warm start to population
        pop += [best_prev] + warm_start_individuals

    else:
        pop = [creator.Individual(toolbox.individual()) for i in range(n_individuals)]

    # Evaluate the entire population
    for ind in pop:
        fit, feasible = toolbox.evaluate(ind)
        ind.fitness.values = (fit,)
        ind.feasible = feasible

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses
    fits = [ind.fitness.values for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    Ymax = []
    Ymin = []
    Yavg = []
    Ystd = []
    X = []

    # The first best individual is the best from the generated population
    bestOfAll = tools.selBest(pop, 1)[0]

    # %% Begin Algorithm
    print("################  Start of evolution  ################")
    while g < generations:
        # NEW GENERATION
        g = g + 1
        X.append(g)
        print(f"---- Generation {g} ----")

        # SELECT THE BEST INDIVIDUALS TO KEEP FOR THE NEXT GENERATION
        if keep_best:
            best_individuals = list(map(toolbox.clone, tools.selBest(pop, keep_best)))

        # SELECTION OF INDIVIDUALS
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # GENETIC OPERATIONS
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # EVALUATE INDIVIDUALS WITH INVALID FITNESS
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            fit, feasible = toolbox.evaluate(ind)
            ind.fitness.values = (fit,)
            ind.feasible = feasible

        print(f"  Evaluated {len(invalid_ind)} individuals")

        # REPLACE POPULATION BY OFFSPRING AND ORDER
        pop[:] = offspring
        pop[:] = tools.selBest(pop, len(pop))

        # REMOVE WORST INDIVIDUALS AND ADD PREVIOUSLY SELECTED BEST INDIVIDUALS TO OFFSPRING
        if keep_best:
            pop[keep_best:] = pop[:-keep_best]
            pop[:keep_best] = best_individuals

        # SUMMARY OF NEW GENERATION
        bestInd = tools.selBest(pop, 1)[0]
        print(
            f"Best individual: {bestInd} \n        Fitness: {bestInd.fitness.wvalues[0]}  Feasible: {bestInd.feasible}")

        worstInd = tools.selWorst(pop, 1)[0]
        print(
            f"Worst individual: {worstInd} \n        Fitness: {worstInd.fitness.wvalues[0]}  Feasible: {worstInd.feasible}")

        print(
            f"Best of all individuals: {bestOfAll} \n        Fitness: {bestOfAll.fitness.wvalues[0]}  Feasible: {bestOfAll.feasible}")

        # STATISTICS
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

        # UPDATE BEST INDIVIDUAL
        if bestInd.fitness.wvalues[0] > bestOfAll.fitness.wvalues[0]:
            bestOfAll = bestInd

    t_end = time.time()
    print("################  End of (successful) evolution  ################")

    # %% SUMMARY OF THE WHOLE EVOLUTION
    best_fitness, best_is_feasible = toolbox.evaluate(bestOfAll)
    best_routes = toolbox.decode(bestOfAll)
    print(f'The best individual {text_feasibility(best_is_feasible)} feasible and its fitness is {-best_fitness}')
    print('Decoding the best individual results in:\n', best_routes)

    # Compare cost to previous cost. If cost is improved, modify previous operation by new operation
    if -best_fitness > -best_fitness_prev:
        print('[IMPORTANT] This optimization results in an improvement of previous optimization... Saving!')
        fleet.set_routes_of_vehicles(best_routes)
        critical_points = {id_ev: (0, ev.state_leaving[0, 0], ev.state_leaving[1, 0], ev.state_leaving[2, 0]) for
                           id_ev, ev in fleet.vehicles.items()}
        fleet.save_operation_xml(path, critical_points, pretty=True)
        report = res.IOTools.OptimizationReport(best_fitness, best_is_feasible, t_end - t_init)
        res.IOTools.save_optimization_report(path, report)

    # Unregister toolbox utilities
    toolbox.unregister('individual')
    toolbox.unregister('evaluate')
    toolbox.unregister('mate')
    toolbox.unregister('mutate')
    toolbox.unregister('select')
    toolbox.unregister('select_worst')
    toolbox.unregister('decode')

    # Check if continue routing
    s = input('DO IT AGAIN? (Y/N)')
    if s == 'N':
        break

# %% If here, there are not more vehicles to route
print('All vehicles reached depot... Finishing!')
sys.exit(1)