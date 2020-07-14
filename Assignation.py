from GA_Assignation import *
from GATools import *
from GA_AlreadyAssigned2 import optimal_route_assignation as improve_route
from GA_AlreadyAssigned2 import individual_from_routes

# %% 1. Specify instance location
# data_folder = 'data/real_data/'
data_folder = 'data/montoya-et-al-2017/adapted/'
# instance_filename = data_folder.split('/')[-2]
capacities = [1, 2, 3, 4]
instances = ['tc1c20s3ct4', 'tc1c20s4ct4', 'tc1c40s8ct1', 'tc1c80s12ct2', 'tc1c160s24ct3']

for instance in instances:
    for cap in capacities:
        instance_filename = instance
        path = f'{data_folder}{instance_filename}.xml'

        print(f'Opening:\n {path}')

        # %% 2. Instance fleet
        fleet = from_xml(path, assign_customers=False)
        fleet.network.draw(save_to=None, width=0.02,
                           edge_color='grey', markeredgecolor='black',
                           markeredgewidth=2.0)[0].show()

        fleet.modify_cs_capacities(cap)
        init_fleet_size = int(
            sum([fleet.network.demand(i) for i in fleet.network.customers]) / fleet.vehicles[0].max_payload) + 2
        fleet.resize_fleet(init_fleet_size)
        fleet.relax_time_windows()

        #input('Ready! Press ENTER to continue...')

        # %% 3. GA hyper-parameters
        CXPB, MUTPB = 0.65, 0.89
        num_individuals = 100
        max_generations = 250
        penalization_constant = 500000
        weights = (0.2, 0.8, 1.2, 0.0, 3.0)  # travel_time, charging_time, energy_consumption, charging_cost
        keep_best = 1  # Keep the 'keep_best' best individuals
        tournament_size = 5
        r = 2

        hyper_parameters = HyperParameters(num_individuals, max_generations, CXPB, MUTPB,
                                           tournament_size=tournament_size,
                                           penalization_constant=penalization_constant,
                                           keep_best=keep_best,
                                           weights=weights,
                                           r=r)

        CXPB, MUTPB = 0.65, 0.85
        num_individuals = 100
        max_generations = 250
        penalization_constant = 500000
        weights = (0.2, 0.8, 1.2, 0.0, 3.0)  # travel_time, charging_time, energy_consumption, charging_cost
        keep_best = 1  # Keep the 'keep_best' best individuals
        tournament_size = 3
        crossover_repeat = 2
        mutation_repeat = 2
        hyper_parameters_improve = HyperParameters(num_individuals, max_generations, CXPB, MUTPB,
                                                   tournament_size=tournament_size,
                                                   penalization_constant=penalization_constant,
                                                   keep_best=keep_best,
                                                   weights=weights,
                                                   crossover_repeat=crossover_repeat,
                                                   mutation_repeat=mutation_repeat)

        # input('Press ENTER to continue...')

    # %% 4. Run algorithm
        done = False
        bestOfAll1 = None
        while not done:
            routes, fleet, bestOfAll1, feasible1, toolbox1, optData1 = optimal_route_assignation(fleet, hyper_parameters,
                                                                                                 data_folder,
                                                                                                 best_ind=bestOfAll1)

            best_improve = individual_from_routes(routes, fleet)
            routes, fleet, bestOfAll2, feasible2, toolbox2, optData2 = improve_route(fleet, hyper_parameters_improve,
                                                                                     data_folder,
                                                                                     best_improve)
            best_fitness, best_is_feasible = toolbox2.evaluate(bestOfAll2)
            print(f'The best individual {"is" if best_is_feasible else "is not"} feasible and its fitness is {-best_fitness}')
            if not best_is_feasible:
                print('INCREASING FLEET SIZE BY 1...')
                fleet.resize_fleet(len(fleet) + 1)
                bestOfAll1.insert(len(fleet.network.customers) + len(fleet) - 2, '|')
                bestOfAll1.append(7*60)
                for i in range(hyper_parameters.r):
                    chg_op = [-1, sample(fleet.network.charging_stations, 1)[0], uniform(10, 20)]
                    index = -len(fleet)
                    bestOfAll1 = bestOfAll1[:index] + chg_op + bestOfAll1[index:]
            else:
                done = True

        best_routes = toolbox2.decode(bestOfAll2)
        print('After decoding:\n', best_routes)

# plot_operation = True if input('Do you want to plot results? (y/n)') == 'y' else False
plot_operation = False

# %% 5. Plot if user wants to
if plot_operation:
    figFitness = figure(plot_width=400, plot_height=300,
                        title='Best fitness evolution')
    figFitness.circle(optData2.generations, np.log(optData2.best_fitness))
    figFitness.xaxis.axis_label = 'Generation'
    figFitness.yaxis.axis_label = 'log(-fitness)'

    # Standard deviation of fitness per generation
    figFitnessStd = figure(plot_width=400, plot_height=300,
                           title='Standard deviation of best fitness per generation')
    figFitnessStd.circle(optData2.generations, optData2.std_fitness)
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
