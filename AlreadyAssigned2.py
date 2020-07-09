# %% Imports
import sys

# Resources
from GA_AlreadyAssigned2 import *

t0 = time.time()

sys.path.append('..')

# %% 1. Specify instance location
# data_folder = 'data/real_data/'
data_folder = 'data/XML_files/60C_2CS_1D_5EV_4CAP_HIGHWEIGHT/02-06-2020_14-17-50_INFEASIBLE_ASSIGNATION/'
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
CXPB, MUTPB = 0.55, 0.65
num_individuals = 120
max_generations = 300
penalization_constant = 500000
weights = (0.2, 0.8, 1.2, 0.0)  # travel_time, charging_time, energy_consumption, charging_cost TODO add waiting times
keep_best = 1  # Keep the 'keep_best' best individuals
tournament_size = 5
crossover_repeat = 1
mutation_repeat = 1

hyper_parameters = HyperParameters(num_individuals, max_generations, CXPB, MUTPB,
                                   tournament_size=tournament_size,
                                   penalization_constant=penalization_constant,
                                   keep_best=keep_best,
                                   weights=weights,
                                   crossover_repeat=crossover_repeat,
                                   mutation_repeat=mutation_repeat)
print(hyper_parameters)

# %% 4. Run algorithm
routes = routes_from_xml(path, fleet)
bestOfAll = individual_from_routes(routes)
routes, fleet, bestOfAll, feasible, toolbox, optData = optimal_route_assignation(fleet, hyper_parameters, data_folder,
                                                                                 best_ind=bestOfAll)

best_fitness, best_is_feasible = toolbox.evaluate(bestOfAll)
best_routes = toolbox.decode(bestOfAll)
print(f'The best individual {"is" if best_is_feasible else "is not"} feasible and its fitness is {-best_fitness}')
print('After decoding:\n', best_routes)

# plot_operation = True if input('Do you want to plot results? (y/n)') == 'y' else False
plot_operation = False

# %% 5. Plot if user wants to
if plot_operation:
    figFitness = figure(plot_width=400, plot_height=300,
                        title='Best fitness evolution')
    figFitness.circle(optData.generations, np.log(optData.best_fitness))
    figFitness.xaxis.axis_label = 'Generation'
    figFitness.yaxis.axis_label = 'log(-fitness)'

    # Standard deviation of fitness per generation
    figFitnessStd = figure(plot_width=400, plot_height=300,
                           title='Standard deviation of best fitness per generation')
    figFitnessStd.circle(optData.generations, optData.std_fitness)
    figFitnessStd.xaxis.axis_label = 'Generation'
    figFitnessStd.yaxis.axis_label = 'Standard deviation of fitness'
    figFitnessStd.left[0].formatter.use_scientific = False

    # Grid
    # p = gridplot([[figFitness, figFitnessStd]], toolbar_location='right')
    # show(p)

    fig, g = fleet.draw_operation(
        color_route=('r', 'b', 'g', 'c', 'y', 'r', 'b', 'g', 'c', 'y', 'r', 'b', 'g', 'c', 'y'),
        save_to=None, width=0.02,
        edge_color='grey', alpha=.1, markeredgecolor='black', markeredgewidth=2.0)
    fig.show()

    figs = fleet.plot_operation_pyplot()
    for k, i in enumerate(figs):
        plt.figure()
        i.tight_layout()
        i.show()
    # fleet.plot_operation()