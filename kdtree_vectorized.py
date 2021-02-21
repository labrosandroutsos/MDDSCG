import pandas as pd
import numpy as np
import timeit
import preprocessing
import random


def make_kd_tree(points, dim, i=0):
    if len(points) > 1:
        points.sort(key=lambda x: x[i])
        i = (i + 1) % dim
        half = len(points) >> 1
        return [
            make_kd_tree(points[: half], dim, i),
            make_kd_tree(points[half + 1:], dim, i),
            points[half]
        ]
    elif len(points) == 1:
        return [None, None, points[0]]


# k nearest neighbors
def get_knn(kd_node, point, k, dim, dist_func, return_distances=True, i=0, heap=None):
    import heapq
    is_root = not heap
    if is_root:
        heap = []
    if kd_node is not None:
        dist = dist_func(point, kd_node[2])
        dx = kd_node[2][i] - point[i]
        if len(heap) < k:
            heapq.heappush(heap, (-dist, kd_node[2]))
        elif dist < -heap[0][0]:
            heapq.heappushpop(heap, (-dist, kd_node[2]))
        i = (i + 1) % dim
        # Searching left branch and then right branch
        for b in [dx < 0] + [dx >= 0] * (dx * dx < -heap[0][0]):
            get_knn(kd_node[b], point, k, dim, dist_func, return_distances, i, heap)
    if is_root:
        neighbors = sorted((-h[0], h[1]) for h in heap)
        return neighbors if return_distances else [n[1] for n in neighbors]


def puts(ll):
    for x in ll:
        print(x)


dim = 3


# Euclidean distance function
def dist_sq(a, b, dim):
    return sum((a[i] - b[i]) ** 2 for i in range(dim))


def dist_sq_dim(a, b):
    return dist_sq(a, b, dim)


# First create kd tree, and then do the knn query.
def kd_and_knn(tree, dimensions, test):
    result1 = []
    start = timeit.default_timer()
    result1.append(tuple(get_knn(tree, [0] * dim, 1000, dim, dist_sq_dim)))
    for t in test:
        result1.append(tuple(get_knn(tree, t, 1000, dim, dist_sq_dim)))
    end = timeit.default_timer() - start
    return end


data1, data2, data3, data4 = preprocessing.KD_tree()
N = 1000
query = data2[0:(N-1)]

start_tree = timeit.default_timer()
kd_tree = make_kd_tree(data2, 3)
end_tree = timeit.default_timer() - start_tree
print("Time for tree construction ", end_tree)
result = 0
for i in range(100):
    result += kd_and_knn(kd_tree, dim, query)

print("Time for knn for 1000 points and k neighbors: ", result/100)
print("\n\n")
