import pandas as pd
import numpy as np
import timeit


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
def kd_and_knn(data1, dimensions, test):
    kd_tree = make_kd_tree(data1, dimensions)
    result1 = []
    start = timeit.default_timer()
    result1.append(tuple(get_knn(kd_tree, [0] * dim, 3, dim, dist_sq_dim)))
    for t in test:
        result1.append(tuple(get_knn(kd_tree, t, 3, dim, dist_sq_dim)))
    end = timeit.default_timer() - start
    return end


# Βάση σεισμών Λάμπρου
Y = pd.read_csv("database.csv")
Y = Y[['Latitude', 'Longitude', 'Magnitude']]
Y = Y.to_numpy()
N = 10
Y1 = Y[np.random.choice(N, size=N, replace=False)]
data = Y.tolist()

query = Y1.tolist()
print("Time for knn: ", kd_and_knn(data, dim, query))
print("\n\n")
