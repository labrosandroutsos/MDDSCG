import csv
import time
import random

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
    print("1 - Search Node")
    print("2 - kNN Query")
    print("-1 - Exit Program")
    print("//////////////////////////////")


choice_data = int(input("0 - Load Dataset\n1 - Random Dataset\n-> "))

if choice_data == 0:
    my_nodes = []

    nodes_counter = 0
    with open("database.csv", mode='r', encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                my_nodes.append(
                    Node([float(row["Latitude"]), float(row["Longitude"]), float(row["Magnitude"])],
                         row["Date"] + " " + row["Time"]))
                nodes_counter += 1
            line_count += 1
        print('Number of Nodes: ' + str(nodes_counter))

else:
    my_nodes = []

    for j in range(0, int(input("Give the number of Nodes you want to create: "))):
        coords = []
        for k in range(0, 3):
            coords.append(random.random() * 1024)
        my_nodes.append(Node(coords, 'data'))

# Build Start
start = time.time()


x_sorted_nodes = sorted(my_nodes, key=lambda l: (l.coords[0], l.coords[1]))
my_root, node_list = create_tree(x_sorted_nodes)
end = time.time()
print("Build Time: " + str(end - start))
# Build End


print_options()
choice = int(input())

while choice != -1:
    # Print Tree
    if choice == 0:
        print('-----------------------')
        start = time.time()
        pre_order(my_root)
        end = time.time()
        print('-----------------------')
    # Search
    elif choice == 1:
        start = time.time()
        knn(my_root, node_list, 3)
        end = time.time()

    print('-----------------------')
    print("Action Time: " + str(end - start))
    print('-----------------------')
    print_options()
    choice = int(input())