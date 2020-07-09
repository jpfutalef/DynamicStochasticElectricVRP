import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = '14'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


def eta(socl, soch, N, tabla: np.ndarray, model: NearestNeighbors):
    socl, soch = socl / 100., soch / 100.

    if 0 <= soch <= 1 and 0 <= soch <= 1:
        if soch > socl:
            SR_m = (soch + socl) / 2.
            SR_l = (soch - socl)
            '''
            tabla = np.array([[0.5, 1, 1.00000000], [0.625, 0.75, 1.00000266], [0.375, 0.75, 1.00001860],
                              [0.75, 0.5, 0.99999203], [0.5, 0.5, 1.00001521], [0.25, 0.5, 1.00002874],
                              [0.875, 0.25, 1.00002146], [0.625, 0.25, 1.00000881], [0.5, 0.25, 1.00000620],
                              [0.375, 0.25, 1.00003347], [0.125, 0.25, 1.00004184]])
            '''
            y = np.array([[SR_m, SR_l]])
            D, IDX = model.kneighbors(y)

            W1 = 1 / (1 + (D[0, 0] / D[0, 1]) + (D[0, 0] / D[0, 2])) if D[0, 1] != 0.0 and D[0, 2] != 0 else 0.0
            W2 = 1 / (1 + (D[0, 1] / D[0, 0]) + (D[0, 1] / D[0, 2])) if D[0, 0] != 0.0 and D[0, 2] != 0 else 0.0
            W3 = 1 / (1 + (D[0, 2] / D[0, 0]) + (D[0, 2] / D[0, 1])) if D[0, 0] != 0.0 and D[0, 1] != 0 else 0.0

            eta_f = W1 * (tabla[IDX[0, 0], 2]) + W2 * (tabla[IDX[0, 1], 2]) + W3 * (tabla[IDX[0, 2], 2])
            eta_bat = (0.8 ** (1 / N)) * eta_f
        else:
            eta_bat = 1
        OUT = eta_bat
    else:
        OUT = 0
    return OUT


if __name__ == '__main__':
    df1 = pd.read_csv('data/real_data/instances_london_bat/21nodes_0_100_1EV/capacity_pu.csv', index_col=0)
    df2 = pd.read_csv('data/real_data/instances_london_bat/21nodes_20_95_1EV/capacity_pu.csv', index_col=0)
    df3 = pd.read_csv('data/real_data/instances_london_bat/21nodes_25_75_1EV/capacity_pu.csv', index_col=0)
    df4 = pd.read_csv('data/real_data/instances_london_bat/21nodes_25_100_1EV/capacity_pu.csv', index_col=0)
    df5 = pd.read_csv('data/real_data/instances_london_bat/21nodes_30_70_1EV/capacity_pu.csv', index_col=0)
    df6 = pd.read_csv('data/real_data/instances_london_bat/21nodes_50_100_1EV/capacity_pu.csv', index_col=0)
    df = pd.concat([df1, df2, df3, df4, df5, df6], axis=1, ignore_index=True)
    df.columns = ['0-100', '20-95', '25,75', '25-100', '30-70', '50-100']

    df.plot(linewidth=1., legend=False)
    plt.axhline(0.8, linestyle='--', color='red', label='Degradación')
    plt.xlabel('Ciclo')
    plt.ylabel('Capacidad [p.u.]')
    plt.legend()
    plt.grid()
    plt.show()

    df1 = pd.read_csv('data/real_data/instances_london_bat/21nodes_0_100_1EV/capacity_pu_end_day.csv', index_col=0)
    df2 = pd.read_csv('data/real_data/instances_london_bat/21nodes_20_95_1EV/capacity_pu_end_day.csv', index_col=0)
    df3 = pd.read_csv('data/real_data/instances_london_bat/21nodes_25_75_1EV/capacity_pu_end_day.csv', index_col=0)
    df4 = pd.read_csv('data/real_data/instances_london_bat/21nodes_25_100_1EV/capacity_pu_end_day.csv', index_col=0)
    df5 = pd.read_csv('data/real_data/instances_london_bat/21nodes_30_70_1EV/capacity_pu_end_day.csv', index_col=0)
    df6 = pd.read_csv('data/real_data/instances_london_bat/21nodes_50_100_1EV/capacity_pu_end_day.csv', index_col=0)
    df = pd.concat([df1, df2, df3, df4, df5, df6], axis=1, ignore_index=True)
    df.columns = ['0-100', '20-95', '25,75', '25-100', '30-70', '50-100']

    df.plot(linewidth=1., legend=False)
    plt.axhline(0.8, linestyle='--', color='red', label='Degradación')
    plt.xlabel('Tiempo [día]')
    plt.ylabel('Capacidad [p.u.]')
    plt.legend()
    plt.grid()
    plt.show()