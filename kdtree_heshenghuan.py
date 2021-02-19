from __future__ import print_function
import math
import operator
from collections import deque
import pandas as pd
import random
import timeit


class KDNode:
    # Class for KD Node with kd-tree data and methods.

    def __init__(self, data=None, parent=None, left=None, right=None,
                 axis=None, sel_axis=None, dimensions=None):

        # New node for a kd-tree.
        # The axis and the sel_axis function are needed for nodes within a tree.
        # parent == None, only when the node is the root node.
        # sel_axis(axis) is used when creating subnodes of the current node. It
        # receives the axis of the parent node and returns the axis of the child
        # node.

        self.data = data
        self.parent = parent
        self.left = left
        self.right = right
        self.axis = axis
        self.sel_axis = sel_axis
        self.dimensions = dimensions

    def children(self):

        # Returns children nodes as (node, position) tuples.
        # Position = 0 means left subnode.
        # Position = 1 means right subnode.

        if self.left and self.left.data is not None:
            yield self.left, 0
        if self.right and self.right.data is not None:
            yield self.right, 1

    def search_node(self, point, k, results, examined, get_dist):

        # k = number of nearest neighbors of point.
        # results = ordered dict, while the key-value pair is (node, distance).
        # examined = set.
        # get_dist = distance function, expecting two points and returning a distance value.

        examined.add(self)

        # best Node
        if not results:
            # results is empty
            bestNode = None
            bestDist = float('inf')
        else:
            # nearest tuple
            bestNode, bestDist = sorted(
                results.items(), key=lambda n_d: n_d[1], reverse=False)[0]

        nodesChanged = False

        # If the current node is closer than the current best, then it becomes the current best.
        nodeDist = get_dist(self)
        if nodeDist < bestDist:
            if len(results) == k and bestNode:
                maxNode, maxDist = sorted(
                    results.items(), key=lambda n: n[1], reverse=True)[0]
                results.pop(maxNode)

            results[self] = nodeDist
            nodesChanged = True
        # If we're equal to the current best, add it.
        elif nodeDist == bestDist:
            results[self] = nodeDist
            nodesChanged = True
        # If we don't have k results yet, add it.
        elif len(results) < k:
            results[self] = nodeDist
            nodesChanged = True

        # If there is a change at the nodes, then find the new best.
        if nodesChanged:
            bestNode, bestDist = sorted(
                results.items(), key=lambda n: n[1], reverse=False)[0]

        # hyperplane that are closer to the search point than the current best.
        # Are there any points on the other side that we split?
        for child, pos in self.children():
            if child in examined:
                continue

            examined.add(child)
            compare, combine = COMPARE_CHILD[pos]

            # Compare the difference between the splitting coordinate of the search point and the current node
            # with the distance from the search point to the current best.
            nodePoint = self.data.get(self.axis, 0.)
            pointPlusDist = combine(point.get(self.axis, 0.), bestDist)
            lineIntersects = compare(pointPlusDist, nodePoint)

            # There can be nearer points on the other side of the plane, so the algorithm must move
            # down the other branch of the tree from the current node looking
            # for closer points, following the same recursive process as the
            # entire search.
            if lineIntersects:
                child.search_node(point, k, results, examined, get_dist)

    def search_knn(self, point, k, dist=None):

        # k nearest neighbors of the given point and their distance from the point.
        # Points must have 3 dimensions.
        # dist is a distance function, expecting two points and returning a
        # distance value.
        # Returns ordered list of (node,distance) tuples/dictionaries..
        prev = None
        current = self

        if dist is None:
            get_dist = lambda n: n.dist(point)
        else:
            get_dist = lambda n: dist(n.data, point)

        # Traverse the tree downwards
        while current:
            if (point.get(current.axis, 0.) <
                    current.data.get(current.axis, 0.)):
                # left subtree
                prev = current
                current = current.left
            else:
                # right subtree
                prev = current
                current = current.right

        if not prev:
            return []

        examined = set()
        results = {}

        # Go to the tree, looking for better solutions
        current = prev
        while current:
            current.search_node(point, k, results, examined, get_dist)
            current = current.parent
        # We sort the final result by the distance in the tuple
        # (<KdNode>, distance).
        return sorted(results.items(), key=lambda a: a[1])

    def __nonzero__(self):
        return self.data is not None

    def __repr__(self):
        return "<%(cls)s - %(data)s>" % dict(cls=self.__class__.__name__,
                                             data=repr(self.data))

    __bool__ = __nonzero__

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.data == other
        else:
            return self.data == other.data

    def __hash__(self):
        return id(self)


# Create a kd-tree
def create(point_list, dimensions, axis=0, sel_axis=None, parent=None):
    # Using a point list, we create a kd tree. Every point in the list must have 3 dimensions. point_list = " " =>
    # empty tree. We have to use the dimensions variable for the tree. Root node is split on the axis.
    # sel_axis(axis) -> when we create subnodes of a node.
    # We use the axis of the parent and we get the axis of the child node as a result.

    if not point_list and not dimensions:
        raise ValueError('either point_list or dimensions must be provided')

    elif point_list:
        dimensions = check_dimensionality(point_list, dimensions)

    # Cycle through the axis
    sel_axis = sel_axis or (lambda prev_axis: (prev_axis + 1) % dimensions)

    if not point_list:
        return KDNode(sel_axis=sel_axis, axis=axis, dimensions=dimensions)

    # Using median to pivot
    point_list.sort(key=lambda point: point.get(axis, 0.))
    median = int(len(point_list) / 2)

    loc = point_list[median]
    root = KDNode(loc, parent, left=None, right=None,
                  axis=axis, sel_axis=sel_axis)
    root.left = create(point_list[:median],
                       dimensions, sel_axis(axis), parent=root)
    root.right = create(point_list[median + 1:],
                        dimensions, sel_axis(axis), parent=root)
    return root


def check_dimensionality(point_list, dimensions):
    #  dimensions = dimensions  # or len(point_list[0])
    for p in point_list:
        if max(p.keys()) > 3:
            raise ValueError(
                'Every point must have 3 dimensions')

    return dimensions


def level_order(tree, include_all=False):
    # Function used for getting the levels of the tree and then visualize at the visualizing function.
    # Returns an iterator over the tree in level-order

    # include_all = True =>  dummy entries at empty nodes and iterator -> inf
    q = deque()
    q.append(tree)
    while q:
        node = q.popleft()
        yield node

        if include_all or node.left:
            q.append(node.left or node.__class__())

        if include_all or node.right:
            q.append(node.right or node.__class__())


def visualizing(tree, max_level=100, node_width=10, left_padding=5):
    # Visualizing the kd tree at the console.

    height = min(max_level, tree.height() - 1)
    max_width = pow(2, height)

    per_level = 1
    in_level = 0
    level = 0

    for node in level_order(tree, include_all=True):

        if in_level == 0:
            print()
            print()
            print(' ' * left_padding, end=' ')

        width = int(max_width * node_width / per_level)

        node_str = (str(node.data) if node else '').center(width)
        print(node_str, end=' ')

        in_level += 1

        if in_level == per_level:
            in_level = 0
            per_level *= 2
            level += 1

        if level > height:
            break
    print()
    print()


# child position => comparison operator
# search left child -> 0
# search right child -> 1


COMPARE_CHILD = {
    0: (operator.le, operator.sub),
    1: (operator.ge, operator.add),
}

# Βάση σεισμών Λάμπρου
Y = pd.read_csv("database.csv")
Y = Y[['Latitude', 'Longitude', 'Magnitude']]
Y = Y.to_numpy()
Y = Y.tolist()

points = []
for i in Y:
    points.append({1: i[0], 2: i[1], 3: i[2]})

# print(points)
# Create the kd tree.
root = create(points, dimensions=3)

N = 10
# random query sample.
query = random.sample(points, N)

# Euclidean distance metric
EuclideanDistance = (lambda a, b: math.sqrt(
    sum(abs(a.get(axis, 0.) - b.get(axis, 0.)) ** 2 for axis in range(len(a)))))
f = EuclideanDistance
result = 0
# Knn query
for i in range(N):
    start_q = timeit.default_timer()
    ans = root.search_knn(query[i], k=3, dist=f)
    end_q = timeit.default_timer() - start_q
    result += end_q

print("Time for knn queries for 10 random points and 3 neighbors: ", result)
# print("visualizing the kd-tree: ")
# visualizing(root)
