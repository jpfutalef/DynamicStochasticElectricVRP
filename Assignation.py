from GA_Assignation import *
from GATools import *

# %% 1. Specify instance location
# data_folder = 'data/real_data/'
data_folder = 'data/XML_files/60C_2CS_1D_5EV_4CAP_HIGHWEIGHT/'
instance_filename = data_folder.split('/')[-2]
# instance_filename = '21nodes_0_100_1EV'
path = f'{data_folder}{instance_filename}.xml'

print(f'Opening:\n {path}')

# %% 2. Instance fleet
fleet = from_xml(path, assign_customers=False)
fleet.network.draw(save_to=None, width=0.02,
                   edge_color='grey', markeredgecolor='black',
                   markeredgewidth=2.0)
# input('Ready! Press ENTER to continue...')

# %% 3. GA hyper-parameters
CXPB, MUTPB = 0.55, 0.65
num_individuals = 2
max_generations = 3
penalization_constant = 500000
weights = (0.2, 0.8, 1.2, 0.0)  # travel_time, charging_time, energy_consumption, charging_cost TODO add waiting times
keep_best = 1  # Keep the 'keep_best' best individuals
tournament_size = 5
r = 4
starting_points = {ev_id: InitialCondition(0, 0, 0, ev.alpha_up, 0) for ev_id, ev in fleet.vehicles.items()}

hyper_parameters = HyperParameters(num_individuals, max_generations, CXPB, MUTPB,
                                   tournament_size=tournament_size,
                                   penalization_constant=penalization_constant,
                                   keep_best=keep_best,
                                   weights=weights,
                                   r=r,
                                   starting_points=starting_points)
print(hyper_parameters)

# input('Press ENTER to continue...')

# %% 4. Run algorithm
bestOfAll = [41, 6, 36, 13, 2, 42, 28, 27, 11, 57, 3, 48, 53, 16, '|', 25, 14, 18, 24, 20, 46, 5, 26, 58, 50, 10, 9, 40, 12, 35, 43, 44, 19, 49, '|', '|', 56, 39, 52, 37, 59, 33, 4, 1, 34, 22, 29, 7, 60, 30, 38, '|', 45, 55, 8, 23, 47, 32, 31, 54, 21, 15, 51, 17, '|', 27, 62, 16.56373656501487, -1, 62, 4.237449588647614, -1, 62, 48.921789529959945, -1, 62, 56.39716194169925, -1, 61, 98.17237120599592, -1, 62, 46.83845016238153, -1, 61, 25.341881873700647, -1, 61, 88.78210160007188, -1, 62, 79.59185040870013, -1, 61, 32.411066594023154, -1, 62, 50.830391646624946, -1, 62, 19.58096643681584, -1, 62, 36.928596757752786, -1, 61, 0.1803300588139738, -1, 62, 53.60377834745351, 33, 61, 2.060448992594422, 30, 61, 13.165070382162792, -1, 62, 0.9064009808382218, -1, 61, 70.74550946214491, -1, 62, 98.92179879936513, 531.9582280013054, 615.6332819892634, 853.5874766065876, 576.9948620430648, 613.8033084713921] # [5, 10, 6, 19, 22, 9, 2, 1, 7, 30, 31, 3, 21, '|', 4, 23, 16, 32, 20, 11, 29, 25, 28, 13, 8, 12, '|', 17, 24, 34, 15, 35, 18, 33, 14, 27, 26, '|', -1, 37, 70.084650769576, -1, 37, 15.74268474653557, -1, 36, 17.59059740917086, -1, 36, 26.360633062144835, -1, 36, 9.140535000073626, -1, 36, 23.640896773191095, -1, 37, 69.99335146084654, -1, 36, 3.3485849028546504, -1, 37, 25.282621645008522, 22, 36, 15.16661207689894, -1, 37, 2.6802198883647685, -1, 36, 21.72651075815043, 554.5411975601453, 571.2431707211375, 680.742173176476]
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
