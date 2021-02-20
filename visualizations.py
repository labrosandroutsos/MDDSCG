from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
import open3d as opend
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
import laspy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from preprocessing import artificial_dataset

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'gshpub.dat')
filename2 = os.path.join(dirname, 'gshpub.csv')
filename3 = os.path.join(dirname, 'Cassette_GT.ply')
filename4 = os.path.join(dirname, 'LDR-EU12_12-2012-137-06.LAS')




def make_txt():
    earth = os.path.join(dirname, 'gshpub.csv')
    santorini = os.path.join(dirname, 'santorini.csv')
    paris = os.path.join(dirname, 'paris.csv')

    with open(santorini, 'r') as inp, open('santorini.txt', 'w') as out:
        for line in inp:
            out.write(line.replace(',', '\t'))

    with open(paris, 'r') as inp, open('paris.txt', 'w') as out:
        for line in inp:
            out.write(line.replace(',', '\t'))

    arti = artificial_dataset()

    print(arti)

    np.savetxt('arti.txt', arti.values, fmt='%f')


def paris_vis():
    cloud = opend.io.read_point_cloud(filename3) # Read the point cloud
    # mesh = opend.io.read_triangle_mesh(filename3)
    # print(mesh)

    downpcd = cloud.voxel_down_sample(voxel_size=0.5)
    # open.io.write_point_cloud(filename4, cloud, write_ascii=True)
    opend.visualization.draw_geometries([downpcd]) # Visualize the point cloud

    downpcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_tree = opend.geometry.KDTreeFlann(downpcd)
    downpcd.colors[1500] = [1, 0, 0]
    [k, idx, _] = pcd_tree.search_knn_vector_3d(downpcd.points[1500], 200)
    np.asarray(downpcd.colors)[idx[1:], :] = [0, 0, 1]

    print(downpcd)
    opend.visualization.draw_geometries([downpcd])

    downpcd.colors = opend.utility.Vector3dVector(np.random.uniform(0, 1, size=(85617, 3)))
    opend.visualization.draw_geometries([downpcd])
    print('octree division')
    octree = opend.geometry.Octree(max_depth=10)
    octree.convert_from_point_cloud(downpcd, size_expand=0.01)
    opend.visualization.draw_geometries([octree])


def earth_vis():
    cloud = opend.io.read_point_cloud(filename, format='xyz') # Read the point cloud
    # mesh = opend.io.read_triangle_mesh(filename3)
    # print(mesh)

    downpcd = cloud.voxel_down_sample(voxel_size=0.25)
    # open.io.write_point_cloud(filename4, cloud, write_ascii=True)
    opend.visualization.draw_geometries([downpcd]) # Visualize the point cloud

    downpcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_tree = opend.geometry.KDTreeFlann(downpcd)
    downpcd.colors[1500] = [1, 0, 0]
    [k, idx, _] = pcd_tree.search_knn_vector_3d(downpcd.points[1500], 200)
    np.asarray(downpcd.colors)[idx[1:], :] = [0, 0, 1]

    print(downpcd)
    opend.visualization.draw_geometries([downpcd])

    downpcd.colors = opend.utility.Vector3dVector(np.random.uniform(0, 1, size=(196638, 3)))
    opend.visualization.draw_geometries([downpcd])
    print('octree division')
    octree = opend.geometry.Octree(max_depth=10)
    octree.convert_from_point_cloud(downpcd, size_expand=0.01)
    opend.visualization.draw_geometries([octree])


def santorini_vis():
    santorini = os.path.join(dirname, 'santorini.txt')

    pcd1 = opend.io.read_point_cloud(santorini, format='xyz')
    print(pcd1)
    # downpcd1 = pcd1.voxel_down_sample(voxel_size=0.5)
    opend.visualization.draw_geometries([pcd1])  # Visualize the point cloud

    downpcd = pcd1.voxel_down_sample(voxel_size=0.05)
    # open.io.write_point_cloud(filename4, cloud, write_ascii=True)
    opend.visualization.draw_geometries([downpcd]) # Visualize the point cloud

    downpcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_tree = opend.geometry.KDTreeFlann(downpcd)
    downpcd.colors[1500] = [1, 0, 0]
    [k, idx, _] = pcd_tree.search_knn_vector_3d(downpcd.points[1500], 200)
    np.asarray(downpcd.colors)[idx[1:], :] = [0, 0, 1]

    print(downpcd)
    opend.visualization.draw_geometries([downpcd])

    downpcd.colors = opend.utility.Vector3dVector(np.random.uniform(0, 1, size=(49230, 3)))
    opend.visualization.draw_geometries([downpcd])
    print('octree division')
    octree = opend.geometry.Octree(max_depth=10)
    octree.convert_from_point_cloud(downpcd, size_expand=0.01)
    opend.visualization.draw_geometries([octree])


def artificial_vis():
    arti = os.path.join(dirname, 'arti.txt')

    cloud = opend.io.read_point_cloud(arti, format='xyz') # Read the point cloud
    # mesh = opend.io.read_triangle_mesh(filename3)
    # print(mesh)

    downpcd = cloud.voxel_down_sample(voxel_size=0.05)
    # open.io.write_point_cloud(filename4, cloud, write_ascii=True)
    opend.visualization.draw_geometries([downpcd]) # Visualize the point cloud

    downpcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_tree = opend.geometry.KDTreeFlann(downpcd)
    downpcd.colors[1500] = [1, 0, 0]
    [k, idx, _] = pcd_tree.search_knn_vector_3d(downpcd.points[1500], 200)
    np.asarray(downpcd.colors)[idx[1:], :] = [0, 0, 1]

    print(downpcd)
    opend.visualization.draw_geometries([downpcd])

    downpcd.colors = opend.utility.Vector3dVector(np.random.uniform(0, 1, size=(9392, 3)))
    opend.visualization.draw_geometries([downpcd])
    print('octree division')
    octree = opend.geometry.Octree(max_depth=10)
    octree.convert_from_point_cloud(downpcd, size_expand=0.01)
    opend.visualization.draw_geometries([octree])

artificial_vis()

def simple_vis():
    santorini = os.path.join(dirname, 'santorini.txt')
    paris = os.path.join(dirname, 'paris.txt')

    pcd = opend.io.read_point_cloud(filename, format='xyz')
    downpcd = pcd.voxel_down_sample(voxel_size=0.5)
    opend.visualization.draw_geometries([downpcd])  # Visualize the point cloud

    pcd1 = opend.io.read_point_cloud(santorini, format='xyz')
    print(pcd1)
    # downpcd1 = pcd1.voxel_down_sample(voxel_size=0.5)
    opend.visualization.draw_geometries([pcd1])  # Visualize the point cloud

    pcd2 = opend.io.read_point_cloud(paris, format='xyz')
    # downpcd2 = pcd2.voxel_down_sample(voxel_size=0.5)
    opend.visualization.draw_geometries([pcd2])  # Visualize the point cloud


