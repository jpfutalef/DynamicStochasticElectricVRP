import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams, rcParamsDefault

from res.models import Fleet

# Pandas options
pd.set_option('display.float_format', '{:.1f}'.format)

# Global parameters
constraints_mapper = {'time_window_down': 'Lower TW', 'time_window_upp': 'Upper TW', 'alpha_up': 'Upper SOC policy',
                      'alpha_down': 'Lower SOC policy', 'max_tour_time': 'Max. tour time', 'cs_capacity': 'CS capacity'}
boxplot_size = (8.5, .8 * 11)  # inches
# boxplot_args = {'notch': True, 'sym': 'o', 'vert': True, 'whis': 1.5, 'meanline': True, 'showmeans': True}
boxplot_args = {}

exec_time_size = (2 * 7.12663125, 2 * 1.826826222223)  # inches


# Useful functions
def one_folder_data(folder: Path, source_folder: Path, fleet: Fleet.Fleet = None):
    instance_name = "\_".join(folder.parent.parent.stem.split('_'))
    simulation_name = "\_".join(folder.stem.split('_'))

    fleet = fleet if fleet else Fleet.from_xml(Path(folder, 'fleet_temp.xml'))
    offline_costs_df = pd.read_csv(Path(source_folder, 'costs.csv'), index_col=0)

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


def folder_data(simulations_folder: Path, source_folder: Path, simulations_number=30,
                data_name='Open Loop + Deterministic DM'):
    costs_table = pd.DataFrame(columns=[])
    constraints_table = pd.DataFrame(columns=[])
    for simulation_folder in simulations_folder.iterdir():
        if simulation_folder.is_file():
            continue
        costs_row, constraints_row = one_folder_data(simulation_folder, source_folder, None)
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
    parser.add_argument('source_folder', type=Path, help='folder containing source files')
    parser.add_argument('--results_folder', type=Path, help='folder where results will be saved')
    parser.add_argument('--number_simulation', type=int, help='num of simulations to consider. Default: 30', default=30)
    args = parser.parse_args()

    target_folder: Path = args.target_folder
    source_folder: Path = args.source_folder

    # Get data
    df_costs, df_constraints = folder_data(target_folder, source_folder)
    df_costs.to_csv(Path(target_folder, 'costs.csv'))
    df_constraints.to_csv(Path(target_folder, 'constraints.csv'))

    characterized_constraints_df = characterize_constraints(df_constraints)

    characterized_constraints_df.to_csv(Path(target_folder, 'constraints_summary.csv'))
    characterized_constraints_df.to_latex(Path(target_folder, 'constraints_summary.tex'),
                                          caption='Summary of constraints violations',
                                          label='tab: constraints violations summary')
