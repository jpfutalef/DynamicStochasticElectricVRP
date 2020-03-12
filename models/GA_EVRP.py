# %%

# Too work with arguments and script paths
import sys

# scientific libraries and utilities
import numpy as np
import pandas as pd
import random
import time
import copy

# GA library
from deap import base
from deap import creator
from deap import tools

# Visualization tools
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_notebook, show
from bokeh.layouts import gridplot, column
from bokeh.layouts import row as layout_row
from bokeh.models.annotations import Arrow, Label
from bokeh.models.arrow_heads import OpenHead, NormalHead, VeeHead
from bokeh.models import ColumnDataSource, Div, Whisker, Span, Range1d
from bokeh.io import export_svgs, export_png

# To reload modules
import importlib

# XML tools
import xml.etree.ElementTree as ET

# EV and network libraries
import res.GA_prev_test
from res.ElectricVehicle import ElectricVehicle
from res.Node import DepotNode, CustomerNode, ChargeStationNode
from res.Network import Network

t0 = time.time()

sys.path.append('..')

# %%
# 1. Specify instance name
instanceName = 'd1c7cs2_ev2'

# 2. Specify file and folder paths
folderPath = '../data/GA_implementation_xml/' + instanceName + '/'
filePath = folderPath + instanceName + '.xml'
print('Opening:', filePath)

# %% 3. Instance Network
net = Network()
net.from_xml(filePath)
# net.draw()

# Usage example
f = 5
t = 6
key = 'energy_consumption'
print(key, net[f][t][key])
# %% 4. Instantiate EVs
vehicles = res.ElectricVehicle.from_xml(filePath, net)

# 5. Proposal of how many customers_per_vehicle each EV will visit
chooseCustomersRandom = False

numVehicles = len(vehicles)
if chooseCustomersRandom:
    ids_customer = copy.deepcopy(net.ids_customer)
    customers_per_car = [int(len(net.ids_customer) / numVehicles)] * numVehicles
    if len(net.ids_customer) % numVehicles != 0:
        customers_per_car[-1] = int(len(net.ids_customer) / numVehicles) + 1

    for i, j in enumerate(customers_per_car):
        print('Car', i, 'must visit', j, 'customer/s')
    print('\n')

    for id_car, num_customers in enumerate(customers_per_car):
        ids_customer_to_visit = []
        for j in range(0, num_customers):
            index = random.randint(0, len(ids_customer) - 1)
            ids_customer_to_visit.append(ids_customer.pop(index))
        print('Car', id_car, 'must visit customers_per_vehicle with ID:', ids_customer_to_visit)
        vehicles[id_car].set_customers_to_visit(ids_customer_to_visit)
else:
    ids_customer = [[1, 2, 3, 4], [5, 6, 7]]
    for id_car, customers in enumerate(ids_customer):
        vehicles[id_car].set_customers_to_visit(customers)

input('Press enter to continue...')

# %%
# 7. GA hyperparameters
CXPB, MUTPB = 0.4, 0.8
n_individuals = 100
generations = 170
penalization_constant = 500000
weights = (1.0, 1.5, 0.0, 0.0)  # travel_time, charging_time, energy_consumption, charging_cost

attrib_ga = {'allowed_charging_operations': 2, 'vehicles': vehicles}
i0, i1, i2 = res.GA_prev_test.createImportantIndices(**attrib_ga)
attrib_ga['indices'] = [i0, i1, i2]
soc_init = 80.0
index = None

# %%
# Fitness objects
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Toolbox
toolbox = base.Toolbox()

toolbox.register("individual", res.GA_prev_test.createRandomIndividual, **attrib_ga)
toolbox.register("evaluate", res.GA_prev_test.fitness, weights=weights,
                 penalization_constant=penalization_constant, x2_0=soc_init, **attrib_ga)
toolbox.register("mate", res.GA_prev_test.crossover, index=None, **attrib_ga)
toolbox.register("mutate", res.GA_prev_test.mutate, index=None, **attrib_ga)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("decode", res.GA_prev_test.decode, **attrib_ga)

# %% the algorithm

tInitGA = time.time()
# Population TODO create function

pop = []
for i in range(0, n_individuals):
    pop.append(creator.Individual(toolbox.individual()))

# Evaluate the entire population
# fitnesses = list(map(toolbox.evaluate, pop))

for ind in pop:
    fit = toolbox.evaluate(ind)
    ind.fitness.values = fit

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
# Begin the evolution
while g < generations:
    # A new generation
    g = g + 1
    X.append(g)
    print("-- Generation %i --" % g)

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
        fit = toolbox.evaluate(ind)
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(invalid_ind))

    # The population is entirely replaced by the offspring
    pop[:] = offspring

    # Gather all the fitnesses in one list and print the stats
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

    bestInd = pop[fits.index(max(fits))]
    print("Best individual: ", bestInd, 'fitness:', sum(bestInd.fitness.wvalues))
    print("Best individual: ", tools.selBest(pop, 1)[0], 'fitness:', sum(tools.selBest(pop, 1)[0].fitness.wvalues))

    worstInd = pop[fits.index(min(fits))]
    print("Worst individual: ", worstInd)

    # Save best ind
    if sum(bestInd.fitness.wvalues) > sum(bestOfAll.fitness.wvalues):
        bestOfAll = bestInd

# %%
print("################  End of (successful) evolution  ################")

# %% Vehicles dynamics
# decode best
S, L, x0 = toolbox.decode(bestOfAll)
print(S, L, x0)

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

# Vectors to plot
colorArrowTravel = 'SteelBlue'
colorArrowCharging = 'Crimson'
colorArrowServing = 'SeaGreen'

# Plot
maxTw = -1
minTw = 100000000000
for vehicleID, vehicle in vehicles.items():
    vehicle: ElectricVehicle
    # assign best sequences
    vehicle.set_sequences(S[vehicleID], L[vehicleID], x0[vehicleID], soc_init)
    reaching_state, leaving_state = vehicle.iterateState()

    # figures
    figX1 = figure(plot_width=600, plot_height=450,
                   title='Time the vehicle ' + str(vehicleID) + 'leaves',
                   toolbar_location='right')
    figX2 = figure(plot_width=600, plot_height=450,
                   title='SOC (vehicle ' + str(vehicleID) + ')',
                   y_range=(0, 100),
                   toolbar_location=None)
    figX3 = figure(plot_width=600, plot_height=450,
                   title='Payload (vehicle ' + str(vehicleID) + ')',
                   toolbar_location=None)

    # time windows
    nSeq = vehicle.node_sequence
    kCustomers = []
    tWindowsUpper = []
    tWindowsLower = []
    for i, node in enumerate(nSeq):
        if net.isCustomer(node):
            kCustomers.append(i)
            node_instance = net.nodes[node]['attr']
            tWindowsCenter = (node_instance.timeWindowUp + node_instance.timeWindowDown) / 2.0
            tWindowsWidth = (node_instance.timeWindowUp - node_instance.timeWindowDown) / 2.0
            tWindowsUpper.append(tWindowsCenter + tWindowsWidth)
            tWindowsLower.append(tWindowsCenter - tWindowsWidth)
            # Time windows whiskers
            whiskerTW = Whisker(base=i, upper=tWindowsCenter + tWindowsWidth, lower=tWindowsCenter - tWindowsWidth)
            figX1.add_layout(whiskerTW)
            # update TW bounds
            if tWindowsCenter + tWindowsWidth > maxTw:
                maxTw = tWindowsCenter + tWindowsWidth
            if tWindowsCenter - tWindowsWidth < minTw:
                minTw = tWindowsCenter - tWindowsWidth

    # adjust fig 1 to fit TWs
    figX1.y_range = Range1d(minTw - 10, maxTw + 10)

    kVehicle = range(0, len(vehicle.node_sequence))

    figX1.line(kVehicle, reaching_state[0, :], alpha=0)
    figX1.line(kVehicle, leaving_state[0, :], alpha=0)
    figX2.line(kVehicle, reaching_state[1, :], alpha=0)
    figX2.line(kVehicle, leaving_state[1, :], alpha=0)
    figX3.line(kVehicle, reaching_state[2, :], alpha=0)

    reaching_vector_prev = reaching_state[:, 0]
    leaving_vector_prev = leaving_state[:, 0]
    nodePrev = nSeq[0]
    kPrev = 0

    label = Label(x=kPrev, y=reaching_vector_prev[0], y_offset=-5, text=str(node), text_baseline='top')
    figX1.add_layout(label)

    label = Label(x=kPrev, y=reaching_vector_prev[1], y_offset=-5, text=str(node), text_baseline='top')
    figX2.add_layout(label)

    label = Label(x=kPrev, y=reaching_vector_prev[2], y_offset=-5, text=str(node), text_baseline='top')
    figX3.add_layout(label)

    # Axes
    figX1.xaxis.axis_label = 'k'
    figX1.yaxis.axis_label = 'Time of the day (min)'
    figX1.axis.axis_label_text_font_size = '15pt'
    figX1.axis.major_label_text_font_size = '13pt'
    figX1.title.text_font_size = '15pt'

    figX2.xaxis.axis_label = 'k'
    figX2.yaxis.axis_label = 'SOC (%)'
    figX2.axis.axis_label_text_font_size = '15pt'
    figX2.axis.major_label_text_font_size = '13pt'
    figX2.title.text_font_size = '15pt'

    figX3.xaxis.axis_label = 'k'
    figX3.yaxis.axis_label = 'Payload (ton)'
    figX3.axis.axis_label_text_font_size = '15pt'
    figX3.axis.major_label_text_font_size = '13pt'
    figX3.title.text_font_size = '15pt'

    # horizontal line SOC
    hline1 = Span(location=40, dimension='width', line_color='black')
    hline2 = Span(location=80, dimension='width', line_color='black')
    figX2.renderers.extend([hline1, hline2])

    for reaching_vector, leaving_vector, node, k in zip(reaching_state[:, 1:].T, leaving_state[:, 1:].T, nSeq[1:],
                                                        range(1, len(vehicle.node_sequence))):
        # x1
        label = Label(x=k, y=reaching_vector[0], y_offset=-5, text=str(node), text_baseline='top')
        figX1.add_layout(label)

        arrowTravel = Arrow(x_start=kPrev, y_start=leaving_vector_prev[0],
                            x_end=k, y_end=reaching_vector[0],
                            end=VeeHead(size=8, fill_color=colorArrowTravel, line_color=colorArrowTravel),
                            line_color=colorArrowTravel, line_alpha=1)

        if net.isChargeStation(node):
            colorArrowSpent = colorArrowCharging
        else:
            colorArrowSpent = colorArrowServing

        arrowSpent = Arrow(x_start=k, y_start=reaching_vector[0],
                           x_end=k, y_end=leaving_vector[0],
                           end=VeeHead(size=8, fill_color=colorArrowSpent, line_color=colorArrowSpent),
                           line_color=colorArrowSpent, line_alpha=1)

        figX1.add_layout(arrowTravel)
        figX1.add_layout(arrowSpent)

        # x2
        label = Label(x=k, y=reaching_vector[1], y_offset=10, text=str(node), text_baseline='top')
        figX2.add_layout(label)

        arrowTravel = Arrow(x_start=kPrev, y_start=leaving_vector_prev[1],
                            x_end=k, y_end=reaching_vector[1],
                            end=VeeHead(size=8, fill_color=colorArrowTravel, line_color=colorArrowTravel),
                            line_color=colorArrowTravel, line_alpha=1, line_width=1.5, visible=True)
        figX2.add_layout(arrowTravel)

        if net.isChargeStation(node):
            colorArrowSpent = colorArrowCharging
            arrowSpent = Arrow(x_start=k, y_start=reaching_vector[1],
                               x_end=k, y_end=leaving_vector[1],
                               end=VeeHead(size=8, fill_color=colorArrowSpent, line_color=colorArrowSpent),
                               line_color=colorArrowSpent, line_alpha=1, line_width=1.5, visible=True)
            figX2.add_layout(arrowSpent)

        # x3
        label = Label(x=k, y=reaching_vector[2], y_offset=-5, text=str(node), text_baseline='top')
        figX3.add_layout(label)

        if net.isChargeStation(node):
            colorArrow = colorArrowCharging

        else:
            colorArrow = colorArrowTravel

        arrowTravel = Arrow(x_start=kPrev, y_start=leaving_vector_prev[2],
                            x_end=k, y_end=leaving_vector[2],
                            end=VeeHead(size=8, fill_color=colorArrow, line_color=colorArrow),
                            line_color=colorArrow, line_alpha=1, line_width=1.5, visible=True)
        figX3.add_layout(arrowTravel)

        # common
        reaching_vector_prev, leaving_vector_prev = reaching_vector, leaving_vector
        nodePrev = node
        kPrev = k

    # Show
    show(figX1)
    time.sleep(0.5)
    show(figX2)
    time.sleep(0.5)
    show(figX3)
    time.sleep(0.5)
