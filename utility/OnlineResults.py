import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams, rcParamsDefault

sys.path.append('..')

import res.models.Fleet as Fleet

pd.set_option('display.float_format', '{:.1f}'.format)

# Global parameters

constraint_names = ['time_window_down', 'time_window_upp', 'alpha_up', 'alpha_down', 'max_tour_time', 'cs_capacity']
constraint_names_pretty = ['Lower TW', 'UpperTW', 'Upper SOC policy', 'Lower SOC policy', 'Max. tour time',
                           'CS capacity']
constraints_mapper = {i: j for i, j in zip(constraint_names, constraint_names_pretty)}

# boxplot_size = (513.11745 / 100, 0.28 * 657.65744 / 100)  # inches
boxplot_size = (8.5, .8*11)  # inches
# boxplot_args = {'notch': True, 'sym': 'o', 'vert': True, 'whis': 1.5, 'meanline': True, 'showmeans': True}
boxplot_args = {}

exec_time_size = (2 * 7.12663125, 2 * 1.826826222223)  # inches


# Useful functions
def one_folder_data(folder: Path, fleet: Fleet.Fleet = None):
    instance_name = "\_".join(folder.parent.parent.stem.split('_'))
    simulation_name = "\_".join(folder.stem.split('_'))

    fleet = fleet if fleet else Fleet.from_xml(Path(folder, 'fleet_temp.xml'))
    offline_costs_df = pd.read_csv(Path(folder.parent.parent, 'source/costs.csv'), index_col=0)

    offline_costs: pd.Series = offline_costs_df.loc['cost']
    weights: pd.Series = offline_costs_df.loc['weight']
    offline_weighted_cost: pd.Series = offline_costs.multiply(weights)

    history_path = Path(folder, 'history.xml')
    history_et: ET.ElementTree = ET.parse(history_path)
    _fleet: ET.Element = history_et.find('fleet')
    costs_dict = {key: float(attr) for key, attr in _fleet.attrib.items()}
    mapper = {'consumed_energy': 'Energy Consumption (SOC)', 'recharging_cost': 'Charging Cost',
              'recharging_time': 'Charging Time (min)', 'travelled_time': 'Travel Time (min)'}
    online_costs = pd.Series(costs_dict, name='cost')
    online_costs.rename(mapper, inplace=True)
    online_costs['Energy Consumption (SOC)'] *= 100. / 24
    if 'Waiting Time (min)' not in online_costs:
        online_costs = online_costs.append(pd.Series(0.0, index=['Waiting Time (min)']))

    online_weighted_costs: pd.Series = online_costs.multiply(weights)
    travel_time = online_costs['Travel Time (min)']
    energy_consumption = online_costs['Energy Consumption (SOC)'] * fleet.vehicles[0].battery_capacity / 100.
    recharging_time = online_costs['Charging Time (min)']
    recharging_cost = online_costs['Charging Cost']
    waiting_time = online_costs['Waiting Time (min)']

    online_cost = online_weighted_costs.sum()

    online_cost_data = {'Operational Cost': online_cost,
                        'J1 (min)': travel_time,
                        'J2 (min)': recharging_time,
                        'J3 (CLP)': recharging_cost,
                        'J4 (kWh)': energy_consumption}

    index = pd.MultiIndex.from_arrays([[instance_name], [simulation_name]],
                                      names=['Instance', 'Optimization Date'])

    cost_data = pd.DataFrame(online_cost_data, index=index)

    # Constraints violation
    constraints_data = constraints_single_folder(folder)

    return cost_data, constraints_data


def folder_data(folder: Path, simulations_number=30):
    data_name = 'Offline' if 'OpenLoop' in folder.stem else 'Offline + Online'
    costs_table = pd.DataFrame(columns=[])
    constraints_table = pd.DataFrame(columns=[])
    for simulation_folder in folder.iterdir():
        if simulation_folder.is_file():
            continue
        costs_row, constraints_row = one_folder_data(simulation_folder, None)
        costs_table = costs_table.append(costs_row)
        constraints_table = constraints_table.append(constraints_row)

    costs_table = costs_table.sort_values('Operational Cost')
    costs_table = costs_table[:simulations_number]
    costs_table.name = data_name
    constraints_table.name = data_name

    return costs_table, constraints_table


# Function to compare costs using boxplots
def boxplot_column_comparison(df1: pd.DataFrame, df2: pd.DataFrame, column: str, xticks=None, **kwargs):
    df = pd.concat([df1[column].reset_index(drop=True), df2[column].reset_index(drop=True)], axis=1, ignore_index=True)
    df.rename(columns={0: df1.name, 1: df2.name}, inplace=True)
    color = {'medians': 'red'}
    meanprops = {'color': 'red', 'linestyle': '--'}
    axes: plt.Axes = df.boxplot(return_type='axes', color=color, meanprops=meanprops, **kwargs)
    axes.set_ylabel(column)
    return axes


def get_execution_times(folder: Path):
    min_executions = np.infty
    max_executions = -np.infty
    data = {}
    counter = 0
    for simulation_folder in folder.iterdir():
        if simulation_folder.is_file():
            continue
        filepath = Path(simulation_folder, 'exec_time.csv')
        execution_time_ser = pd.read_csv(filepath, index_col=0, squeeze=True)
        column_name = f'Simulation {counter}'
        column_data = execution_time_ser.values
        column_length = len(column_data)
        data[column_name] = column_data
        if column_length > max_executions:
            max_executions = column_length
        if column_length < min_executions:
            min_executions = column_length
        counter += 1

    index = pd.Index(data=range(min_executions), name='Execution number')
    full_df = pd.DataFrame({i: j[:min_executions] for i, j in data.items()}, index=index)
    return full_df, min_executions, max_executions


def constraints_single_folder(folder: Path):
    history_filepath = Path(folder, 'history.xml')
    # columns = ['time_window_down', 'time_window_upp', 'alpha_up', 'alpha_down', 'max_tour_time', 'cs_capacity', 'total']
    simulation_name = folder.stem
    if '_' in simulation_name:
        i = simulation_name.index('_')
        simulation_name = simulation_name[:i] + '\_' + simulation_name[i + 1:]

    tree_root = ET.parse(history_filepath).getroot()
    columns = ['type', 'where', 'when', 'constraint_value', 'real_value']
    header = ['Constraint', 'Where', 'When', 'Desired value', 'Obtained value']
    column_mapper = {i: j for i, j in zip(columns, header)}
    data = {}

    for _element in tree_root:
        _element: ET.Element
        the_element = _element.tag
        the_id = _element.get('id') if the_element == 'vehicle' else ''
        for _violated_constraint in _element.find('violated_constraints'):
            _violated_constraint: ET.Element
            where = int(_violated_constraint.get('where'))
            when = str(_violated_constraint.get('when'))
            the_type = str(_violated_constraint.get('type'))
            real_value = float(_violated_constraint.get('real_value'))
            constraint_value = float(_violated_constraint.get('constraint_value'))
            diff = abs(real_value - constraint_value)
            index = (simulation_name, the_element + the_id, the_type, where, when)
            data[index] = [constraint_value, real_value, diff]

    if data:
        df = pd.DataFrame(data).T.set_axis(['Constraint bound', 'Obtained value', 'Difference'], axis=1)
        df.index.set_names(['Simulation folder', 'Violation source', 'Violated constraint', 'Where', 'When'],
                           inplace=True)
    else:
        df = pd.DataFrame(columns=['Constraint bound', 'Obtained value', 'Difference'])
    return df


def characterize_constraints(df: pd.DataFrame, constraints_pretty: list = None):
    constraints = ['time_window_down', 'time_window_upp', 'alpha_down', 'alpha_up', 'max_tour_time',
                   'cs_capacity']
    if constraints_pretty is None:
        constraints_pretty = ['Time window lower bound', 'Time window upper bound',
                              'SOC policy lower bound', 'SOC policy upper bound', 'Maximum tour time',
                              'CS capacity']

    group = df['Difference'].groupby(level='Violated constraint')
    description = group.describe()

    for i in constraints:
        if i in description.index:
            continue
        row_data = [0.] * len(description.columns)
        row = pd.Series(row_data, index=description.columns, name=i)
        description = description.append(row)
    description['count'] = description['count'].astype(int)
    description = description.rename({i: j for i, j in zip(constraints, constraints_pretty)}).sort_index()
    header = pd.MultiIndex.from_product([[df.name], description.columns])
    description.set_axis(header, axis=1, inplace=True)
    description.drop(['25%', '50%', '75%'], axis=1, level=1, inplace=True)
    description.fillna(0., inplace=True)
    return description


if __name__ == '__main__':
    rcParams.update(rcParamsDefault)
    # rcParams['font.size'] = '10'
    # rcParams['pdf.fonttype'] = 42

    parser = argparse.ArgumentParser(description='Obtains the results from online operation')
    parser.add_argument('target_folder', type=Path, help='specifies target folder')
    parser.add_argument('--results_folder', type=Path, help='folder where results will be saved')
    parser.add_argument('--number_simulation', type=int, help='num of simulations to consider. Default: 30', default=30)
    args = parser.parse_args()

    target_folder: Path = args.target_folder

    # Get data
    off_df, off_constraints_df = folder_data(Path(target_folder, 'simulations_OpenLoop'), args.number_simulation)
    on_df, on_constraints_df = folder_data(Path(target_folder, 'simulations_ClosedLoop'), args.number_simulation)

    off_df.to_csv(Path(target_folder, 'costs_OpenLoop.csv'))
    on_df.to_csv(Path(target_folder, 'costs_ClosedLoop.csv'))

    off_constraints_df.to_csv(Path(target_folder, 'constraints_OpenLoop.csv'))
    on_constraints_df.to_csv(Path(target_folder, 'constraints_ClosedLoop.csv'))

    characterized_off_constraints_df = characterize_constraints(off_constraints_df)
    characterized_on_constraints_df = characterize_constraints(on_constraints_df)

    constraints_summary = characterized_off_constraints_df.join(characterized_on_constraints_df)
    constraints_summary.to_csv(Path(target_folder, 'constraints_summary.csv'))
    constraints_summary.to_latex(Path(target_folder, 'constraints_summary.tex'),
                                 caption='Summary of constraints violations',
                                 label='tab: constraints violations summary')

    # Boxplot comparison among costs
    fig = plt.figure(figsize=boxplot_size)
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    ax5 = plt.subplot2grid((3, 2), (2, 1))

    boxplot_column_comparison(off_df, on_df, 'Operational Cost', ax=ax1, **boxplot_args)
    boxplot_column_comparison(off_df, on_df, 'J1 (min)', ax=ax2, **boxplot_args)
    boxplot_column_comparison(off_df, on_df, 'J2 (min)', ax=ax3, **boxplot_args)
    boxplot_column_comparison(off_df, on_df, 'J3 (CLP)', ax=ax4, **boxplot_args)
    boxplot_column_comparison(off_df, on_df, 'J4 (kWh)', ax=ax5, **boxplot_args)
    fig.tight_layout()

    fig.savefig(Path(target_folder, 'boxplot_comparison.pdf'))

    # OnGA execution time
    exec_times_df, min_exec, max_exec = get_execution_times(Path(target_folder, 'simulations_ClosedLoop'))
    exec_times_df.to_csv(Path(target_folder, 'OnGA_Full.csv'))
    pd.Series({'Minimum OnGA executions': min_exec,
               'Maximum OnGA executions': max_exec}).to_csv(Path(target_folder, 'OnGA_numOfExecutions.csv'))

    exec_times_mean = exec_times_df.mean(1)
    exec_times_mean.name = 'Mean'
    exec_times_std = exec_times_df.std(1)
    pd.DataFrame({'Mean': exec_times_mean,
                  'Standard Deviation': exec_times_std},
                 index=exec_times_df.index).to_csv(Path(target_folder, 'OnGA_ExecutionTimes.csv'))

    fig, ax = plt.subplots(figsize=exec_time_size)
    x = range(min_exec)
    y0 = exec_times_mean + 3 * exec_times_std
    y1 = exec_times_mean - 3 * exec_times_std

    ax.fill_between(x, y0.values, y1.values, color='k', alpha=.2, interpolate=False, label='99.7% confidence interval')
    exec_times_mean.plot(ax=ax, color='black', )

    ax.legend()
    ax.set_xlabel(r'Measurement instant $\bar k$')
    ax.set_ylabel('Execution time [s]')
    ax.grid(axis='y')
    ax.set_title('OnGA execution time')

    fig.tight_layout()
    fig.savefig(Path(target_folder, 'OnGA_executionTime.pdf'))
