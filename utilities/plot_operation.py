import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from Fleet import routes_from_xml
#rcParams['font.family'] = 'Times New Roman'
#rcParams['font.size'] = '12'
#rcParams['pdf.fonttype'] = 42
#rcParams['ps.fonttype'] =

import matplotlib as mpl
mpl.use('Qt5Agg')

sys.path.append('..')
from models.Fleet import from_xml

instance = 'data/instances/c10cs1_40x40km.xml'
opt_res_folder = 'data/instances/c10cs1_40x40km/4/16-07-2020_01-54-48_INFEASIBLE_ASSIGNATION/'

fleet = from_xml(instance)

routes = routes_from_xml(f'{opt_res_folder}/assigned.xml', fleet)
fleet.resize_fleet(len(routes))

'''
fleet_size = len(fleet.vehicles)
routes = {}
for i in range(fleet_size):
    ev_file = f'{opt_res_folder}/EV{i}_operation.csv'
    df = pd.read_csv(ev_file)
    Sk = tuple(df['Sk'])
    Lk = tuple(df['Lk'])
    x1_0 = df['x1_reaching'].iloc[0]
    x2_0 = df['x2_reaching'].iloc[0]
    x3_0 = df['x3_reaching'].iloc[0]
    routes[i] = ((Sk, Lk), x1_0, x2_0, x3_0)
'''

fleet.set_routes_of_vehicles(routes)
fleet.create_optimization_vector()

plots = fleet.plot_operation_pyplot(label_offset=(.05, -7), fig_size=(16, 4), save_to=opt_res_folder+'/')
plt.show()

#%%
fleet.plot_operation()

#%%

folder = '../data/GA_implementation_xml/c10cs1_20x20km/'
folder_opt = '16-07-2020_01-14-19_FEASIBLE_ASSIGNATION/'
file_path = folder + folder_opt + 'nodes_occupation.csv'
fig_path = folder + folder_opt + 'occupation1_legend.pdf'
df = pd.read_csv(file_path)
plot = df[['36', '37']].plot(yticks=[0,1,2], figsize=(3.2,3), legend=False)
plt.grid(axis='y')
plt.xlabel(r'Event $\bar{k}$')
plt.ylabel(r'Number of EVs')
plt.title('Max. capacity = 3')
plt.legend(('CS 36', 'CS 37'), loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(fig_path)

#%%

fleet.draw_operation(color_route=('r', 'b', 'g', 'c', 'y'), save_to=None, width=0.02, edge_color='grey',
                     markeredgecolor='black', markeredgewidth=2.0)

