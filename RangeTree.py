import math
import sys
#Build:root, node_list = create_tree(nodes)
#knn:knn(roo, node_list, k)

class Node:
    def __init__(self, coords, data=None, left=None, right=None, bst=None):
        self.coords = coords
        self.data = data
        self.left = left
        self.right = right
        self.next_dimension = bst


# returns the tree root (type Node) and a list (used for next dimension trees)
def create_tree(my_list, dimension=0):
    if len(my_list) == 0 or dimension >= 3:
        return None, []

    mid = int(len(my_list) / 2)

    root = my_list[mid]
    root.left, left_list = create_tree(my_list[:mid], dimension)
    root.right, right_list = create_tree(my_list[mid + 1:], dimension)

    merged_list = []
    if dimension + 1 < 3:
        merged_list = merge(root, left_list, right_list, dimension + 1)

    root.next_dimension, _ = create_tree(merged_list, dimension + 1)

    return root, merged_list


# arguments: root and two sorted lists
# returns one sorted list
def merge(root, left_list, right_list, dimension=0):
    if dimension >= 3:
        return []

    final_list = []

    left_index = 0
    right_index = 0

    while left_index < len(left_list) and right_index < len(right_list):
        if left_list[left_index].coords[dimension] < right_list[right_index].coords[dimension]:
            final_list.append(Node(left_list[left_index].coords, left_list[left_index].data))
            left_index = left_index + 1
        else:
            final_list.append(Node(right_list[right_index].coords, right_list[right_index].data))
            right_index = right_index + 1

    while left_index < len(left_list):
        final_list.append(Node(left_list[left_index].coords, left_list[left_index].data))
        left_index = left_index + 1

    while right_index < len(right_list):
        final_list.append(Node(right_list[right_index].coords, right_list[right_index].data))
        right_index = right_index + 1

    if root is None:
        return final_list

    for i in range(0, len(final_list)):
        if root.coords[dimension] < final_list[i].coords[dimension]:
            # case middle
            return final_list[: i] + [Node(root.coords, root.data)] + final_list[i:]

    # case last
    return final_list + [Node(root.coords, root.data)]


# returns type Node
def leftmost_node(node):
    while node.left is not None:
        node = node.left
    return node


def knn(root, node_list, k):
    coords_array = []
    d1 = []
    d2 = []
    d3 = []
    for x in node_list:
        coords_array.append(x.coords)

    for y in coords_array:
        d1.append(y[0])
        d2.append(y[1])
        d3.append(y[2])

    for z in coords_array:
        range = []
        e1 = binary_search(z, d1, 0, len(d1) - 1, 0, 0)  #dimensions(0,1,2)
        e2 = binary_search(z, d2, 0, len(d2) - 1, 1, 0)
        e3 = binary_search(z, d3, 0, len(d3) - 1, 2, 0)
        e = min(e1, e2, e3)
        limit1 = [z[0] - e, z[1] - e, z[2] - e]
        limit2 = [z[0] + e, z[1] + e, z[2] + e]
        limit3 = [z[0] + e/2, z[1] + e/2, z[2] + e/2]
        range.append(limit1)
        range.append(limit2)
        range.append(limit3)
        bruteforce(range_search(root, range), k)


def binary_search(q, coord, low, high, dimension, e):
    # Check base case

    if high >= low:

        mid = (high + low) // 2
        pmid = [coord[mid]]
        qpoint = [q[dimension]]
        # If element is present at the middle itself
        if coord[mid] == q[dimension]: #we found it, we need the distance before this
            return e

        # If element is smaller than mid, then it can only
        # be present in left subarray

        elif coord[mid] > q[dimension]:
            e = math.dist(pmid, qpoint)
            return binary_search(q, coord, low, mid - 1, dimension, e)

        # Else the element can only be present in right subarray
        else:
            e = math.dist(pmid, qpoint)
            return binary_search(q, coord, mid + 1, high, dimension, e)
    else:
        return sys.maxsize;#maximum value


def bruteforce(rng, k):
    coords_array = []
    for x in rng:
        if x.coords:
            coords_array.append(x.coords)
    return coords_array[:k]

# returns the Node where the split takes place
def find_split_node(root, range_coords, dimension=0):
    if root:
        if root.coords[dimension] < range_coords[dimension][0]:
            return find_split_node(root.right, range_coords, dimension)
        elif root.coords[dimension] > range_coords[dimension][1]:
            return find_split_node(root.left, range_coords, dimension)
        else:
            # root is the split node or None
            return root

    return None


# checks if coords are inside a given range
def is_in_range(coords, range_coords):
    for d in range(0, 3):
        if coords[d] < range_coords[d][0] or coords[d] > range_coords[d][1]:
            return False

    return True


# returns a list of Nodes inside a range
def range_search(root, range_coords, dimension=0):
    if root is None:
        return []
    # last dimension
    if dimension + 1 == 3:
        if root.coords[dimension] < range_coords[dimension][0]:
            return range_search(root.right, range_coords, dimension)
        elif root.coords[dimension] > range_coords[dimension][1]:
            return range_search(root.left, range_coords, dimension)
        else:
            nodes_list = [root] + range_search(root.right, range_coords, dimension) + range_search(root.left,
                                                                                                   range_coords,
                                                                                                   dimension)

            return nodes_list

    # d-1 dimensions
    if dimension + 1 < 3:
        split_node = find_split_node(root, range_coords, dimension)

        if split_node is None:
            return []

        nodes_list = []
        if is_in_range(split_node.coords, range_coords):
            nodes_list.append(split_node)

        left_child = split_node.left
        while left_child:
            if is_in_range(left_child.coords, range_coords):
                nodes_list.append(left_child)
            if range_coords[dimension][0] <= left_child.coords[dimension]:
                # 1DRangeSearch
                if left_child.right:
                    nodes_list += range_search(left_child.right.next_dimension, range_coords,
                                               dimension + 1)  # go to the next dimension
                left_child = left_child.left  # continue same dimension
            else:
                left_child = left_child.right  # continue same dimension

        right_child = split_node.right
        while right_child:
            if is_in_range(right_child.coords, range_coords):
                nodes_list.append(right_child)
            if right_child.coords[dimension] <= range_coords[dimension][1]:
                # 1DRangeSearch
                if right_child.left:
                    nodes_list += range_search(right_child.left.next_dimension, range_coords,
                                               dimension + 1)  # go to the next dimension
                right_child = right_child.right  # continue same dimension
            else:
                right_child = right_child.left  # continue same dimension
        return nodes_list



