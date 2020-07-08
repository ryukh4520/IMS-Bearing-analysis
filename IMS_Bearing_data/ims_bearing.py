import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# path
dataset_path = ''
csv_path = '/merged_meanabs.csv'
nor_csv_path = '/normalized_meanabs.csv'


# merge  files
def merge_data(dataset_path):
    nfeature = 8
    column_names = ['Bearing 1-x', 'Bearing 1-y', 'Bearing 2-x', 'Bearing 2-y',
                    'Bearing 3-x', 'Bearing 3-y', 'Bearing 4-x', 'Bearing 4-y']

    merged_data = pd.DataFrame()
    for filename in os.listdir(dataset_path):
        dataset = pd.read_csv(os.path.join(dataset_path, filename), sep='\t')
        dataset_mean_abs = np.array(dataset.abs().mean())
        dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1, nfeature))
        dataset_mean_abs.index = [filename]
        merged_data = merged_data.append(dataset_mean_abs)
    merged_data.columns = column_names
    merged_data.index = pd.to_datetime(merged_data.index, format='%Y.%m.%d.%H.%M.%S')
    merged_data = merged_data.sort_index()
    merged_data.to_csv('merged_meanabs.csv')

    # min_max normalize
    nor_data = merged_data.loc[:, 'Bearing 1-x':'Bearing 4-y']
    min_max_scaler = MinMaxScaler()

    nor_data = min_max_scaler.fit_transform(nor_data)
    nor_merged_data = pd.DataFrame(nor_data, columns=column_names, index=merged_data.index)
    nor_merged_data.to_csv('normalized_meanabs.csv')


# split data per each bearing
def split_bearing():
    df = pd.read_csv(csv_path)
    df_nor = pd.read_csv(nor_csv_path)

    for i in range(1, 5):
        df_s = df.loc[:,f'Bearing {i}-x':f'Bearing {i}-y']
        df_s.to_csv(f'meanabs_b{i}.csv', index=False)

        df_ns = df_nor.loc[:,f'Bearing {i}-x':f'Bearing {i}-y']
        df_ns.to_csv(f'nor_meanabs_b{i}.csv', index=False)


# clustering
def bearing_kmeans(n):
    df_n = pd.read_csv(nor_csv_path)
    df_nl = df_n.loc[:, f'Bearing {n}-x':f'Bearing {n}-y']
    model = KMeans(n_clusters=2, init="k-means++").fit(df_nl)
    c0, c1 = model.cluster_centers_
    label = model.fit_predict(df_nl)

    df_nl["Label"] = label
    df_nl.to_csv(f'nor_meanabs_b{n}_labeled.csv', index=False)


"""
# K-means scatter plot
    plt.scatter(df_n[f'Bearing {n}-x'], df_n[f'Bearing {n}-y'], marker=".", edgecolors='w', s=90, c=label, alpha=0.8)
    plt.scatter(c0[0], c0[1], marker='v', c="r", s=200)
    plt.scatter(c1[0], c1[1], marker='^', c="y", s=200)
    plt.title(f'Bearing {n}, normalized K-means clustering')
    plt.xlabel(f"X-axis normalized vibration")
    plt.ylabel(f"Y-axis normalized vibration")
    plt.show()

"""

merge_data(dataset_path)
split_bearing()

for i in range (1,5):
    bearing_kmeans(i)

