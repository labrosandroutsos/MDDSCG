import csv
import timeit
import random
import preprocessing

def pre_order(root, string=""):
    if root:
        print(string + str(root.coords) + "|data:" + str(root.data))

        pre_order(root.left, "\t" + string + "-left-")
        pre_order(root.right, "\t" + string + "-right-")


def print_nodes(nodes_list):
    for node in nodes_list:
        print(str(node.coords) + "\t|\tdata:" + str(node.data))

# File to import selection
choice = int(input("1 - Range\n2 - Range with Fractional Cascading\n-> "))

if choice == 1:
    from RangeTree import *
else:
    from FCRangeTree import *


def print_options():
    print("\n//////////// MENU ////////////")
    print("0 - Print Tree")
    print("1 - kNN Query")
    print("-1 - Exit Program")
    print("//////////////////////////////")


print("Which database do you want?")
print("1 for Earthquake Data")
print("2 for Santorini Lidar Data")
print("3 for Paris Static Scanner Data")
print("4 for Artificial Data")

choose_data = int(input("Enter a number from above: "))
data1, data2, data3, data4 = preprocessing.KD_tree()
my_nodes = []


def load_data(selection):
    if selection == 1:
        data = data1
    elif selection == 2:
        data = data2
    elif selection == 3:
        data = data3
    else:
        data = data4
    nodes_counter = 0
    el_count = 0
    for el in data:
        my_nodes.append(Node([float(el[0]), float(el[1]), float(el[2])], el_count))
        nodes_counter += 1
        el_count += 1
    print('Number of Nodes: ' + str(nodes_counter))


load_data(choose_data)

# Build Start
start = timeit.default_timer()


x_sorted_nodes = sorted(my_nodes, key=lambda l: (l.coords[0], l.coords[1]))
my_root, node_list = create_tree(x_sorted_nodes)
end = timeit.default_timer() - start
print("Build Time: " + str(end))
# Build End


print_options()
choice = int(input())

while choice != -1:
    # Print Tree
    if choice == 0:
        print('-----------------------')
        start = timeit.default_timer()
        pre_order(my_root)
        end = timeit.default_timer() - start
        print('-----------------------')
    # Search
    elif choice == 1:
        start = timeit.default_timer()
        knn(my_root, node_list, 3)
        end = timeit.default_timer() - start

    print('-----------------------')
    print("Action Time: " + str(end))
    print('-----------------------')
    print_options()
    choice = int(input())
