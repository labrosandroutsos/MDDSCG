from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
import numpy as np
import open3d as open
import warnings
warnings.filterwarnings('ignore')
import os
import plotly.graph_objs as go
from plotly.offline import plot
import h5py
from plyfile import PlyData, PlyElement
from laspy.file import File
from pyntcloud import PyntCloud
import csv

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'gshpub.dat')
filename2 = os.path.join(dirname, 'gshpub.csv')
filename3 = os.path.join(dirname, 'Cassette_GT.ply')
filename4 = os.path.join(dirname, 'LDR-EU12_12-2012-137-06.LAS')

# with open(".csv", mode='r', encoding="utf-8") as csv_file:
#     csv_reader = csv.DictReader
#
#



def las_to_csv():
    inFile = File(filename4, mode = "r")

    # pointformat = inFile.point_format
    # for spec in inFile.point_format:
    #     print(spec.name)

    X = inFile.X
    Y = inFile.Y
    Z = inFile.Z

    combined = np.column_stack((X,Y,Z))
    df = pd.DataFrame(combined)

    df.to_csv (r'C:\Users\A. Androutsopoulos\PycharmProjects\MDDSCG\santorini.csv', index=None)



def dat_to_csv():

    data = pd.read_csv(filename, sep='\s+', header=None)
    data = data.dropna()
    data.to_csv (r'C:\Users\A. Androutsopoulos\PycharmProjects\MDDSCG\gshpub.csv', index=None)

def ply_to_csv():

    plydata = PlyData.read(filename3)

    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']

    combined = np.column_stack((x,y,z))
    df = pd.DataFrame(combined)
    # print(df)

    df.to_csv (r'C:\Users\A. Androutsopoulos\PycharmProjects\MDDSCG\paris.csv', index=None)

def ply_cloud_vis():
    cloud = open.io.read_point_cloud(filename3) # Read the point cloud
    # open.io.write_point_cloud(filename4, cloud, write_ascii=True)
    open.visualization.draw_geometries([cloud]) # Visualize the point cloud

def visualize_3d(X, y, algorithm="tsne", title="Data in 3D"):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    if algorithm == "tsne":
        reducer = TSNE(n_components=3, random_state=47, n_iter=400, angle=0.6)
    elif algorithm == "pca":
        reducer = PCA(n_components=3, random_state=47)
    else:
        raise ValueError("Unsupported dimensionality reduction algorithm given.")

    if X.shape[1] > 3:
        X = reducer.fit_transform(X)
    else:
        if type(X) == pd.DataFrame:
            X = X.values

    marker_shapes = ["circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open", ]
    traces = []
    for hue in np.unique(y):
        X1 = X[y == hue]

        trace = go.Scatter3d(
            x=X1[:, 0],
            y=X1[:, 1],
            z=X1[:, 2],
            mode='markers',
            name=str(hue),
            marker=dict(
                size=12,
                symbol=marker_shapes.pop(),
                line=dict(
                    width=int(np.random.randint(3, 10) / 10)
                ),
                opacity=int(np.random.randint(6, 10) / 10)
            )
        )
        traces.append(trace)

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title='Dim 1'),
            yaxis=dict(
                title='Dim 2'),
            zaxis=dict(
                title='Dim 3'), ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=traces, layout=layout)
    plot(fig)


def artificial_dataset():

    X, y = make_classification(n_samples=10000, n_features=3, n_informative=3,
                               n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=2,
                               class_sep=1.5,
                               flip_y=0, weights=[0.5, 0.5, 0.5])
    X = pd.DataFrame(X)
    y = pd.Series(y)

    visualize_3d(X, y)
    return X


def read_dataset():
    dirname = os.path.dirname(__file__)
    earth = os.path.join(dirname, 'gshpub.csv')
    santorini = os.path.join(dirname, 'santorini.csv')
    paris = os.path.join(dirname, 'paris.csv')

    earthquakes_prob = pd.read_csv(earth)
    santorini_lidar = pd.read_csv(santorini)
    paris_static_scanner = pd.read_csv(paris)
    artidicial_data = artificial_dataset()

    return earthquakes_prob, santorini_lidar, paris_static_scanner, artidicial_data


earthquakes_prob, santorini_lidar, paris_static_scanner, artificial_data = read_dataset()

def KD_tree():

    earthquakes = earthquakes_prob.to_numpy()
    earthquakes = earthquakes.tolist()

    santorini = santorini_lidar.to_numpy()
    santorini = santorini.tolist()

    paris = paris_static_scanner.to_numpy()
    paris = paris.tolist()

    artificial = artificial_data.to_numpy()
    artificial = artificial.tolist()

    return earthquakes,santorini, paris, artificial
# ply_cloud_vis()

