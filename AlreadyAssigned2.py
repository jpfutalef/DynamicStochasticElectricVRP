# %% Imports
import sys

# Resources
from GA_AlreadyAssigned2 import *

t0 = time.time()

sys.path.append('..')

# %% 1. Specify instance location
data_folder = 'data/instances/c25cs3_20x20km/2/15-07-2020_21-36-25_FEASIBLE_ASSIGNED/'
instance_filename = 'assigned'
path = f'{data_folder}{instance_filename}.xml'

print(f'Opening:\n {path}')

# %% 2. Instance fleet
fleet = from_xml(path, assign_customers=True)
fleet.network.draw(save_to=None, width=0.02,
                   edge_color='grey', markeredgecolor='black',
                   markeredgewidth=2.0)
#input('Press enter to continue...')

# %% 3. GA hyperparameters
CXPB, MUTPB = 0.65, 0.85
num_individuals = 1
max_generations = 3
penalization_constant = 500000
weights = (0.2, 0.8, 1.2, 0.0, 3.0)  # travel_time, charging_time, energy_consumption, charging_cost
keep_best = 1  # Keep the 'keep_best' best individuals
tournament_size = 3
crossover_repeat = 2
mutation_repeat = 2

hyper_parameters = HyperParameters(num_individuals, max_generations, CXPB, MUTPB,
                                   tournament_size=tournament_size,
                                   penalization_constant=penalization_constant,
                                   keep_best=keep_best,
                                   weights=weights,
                                   crossover_repeat=crossover_repeat,
                                   mutation_repeat=mutation_repeat)
print(hyper_parameters)

#input('Press enter to continue...')
# %% 4. Run algorithm
routes = routes_from_xml(path, fleet)
bestOfAll = [15, 7, 1, 4, 23, 18, 14, 3, 19, 13, 16, -1, 20.6936038509605, -1, 6.022805206112359, -1, 22.011423704726468, -1, 5.43706562351569, -1, 9.435520364850387, 28, 23.656725396760784, 28, 37.16365407589693, -1, 12.410515861751833, -1, 17.556727267751036, -1, 7.073503972199438, -1, 18.686691490711823, 72.91240035996373, 11, 17, 22, 10, 20, -1, 5.114038661178393, -1, 19.2089648925821, -1, 19.10405355295757, -1, 14.623635278259977, -1, 7.592220512037036, 90.29427624930973, 24, 21, 8, 6, 12, 9, 5, 25, 2, -1, 12.947251990845054, -1, 13.58844484882668, -1, 7.82714716325974, 28, 30.57495997836717, -1, 0.4110719448640765, -1, 18.70194850864131, -1, 29.903732306255154, 28, 28.511965096890542, -1, 20.313587392976327, 158.41569366938253]#individual_from_routes(routes, fleet)
routes, fleet, bestOfAll, feasible, toolbox, optData = optimal_route_assignation(fleet, hyper_parameters, data_folder,
                                                                                 best_ind=bestOfAll)

best_fitness, best_is_feasible = toolbox.evaluate(bestOfAll)
best_routes = toolbox.decode(bestOfAll)
print(f'The best individual {"is" if best_is_feasible else "is not"} feasible and its fitness is {-best_fitness}')
print('After decoding:\n', best_routes)

plot_operation = True if input('Do you want to plot results? (y/n)') == 'y' else False
# plot_operation = False

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