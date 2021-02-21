import timeit
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import preprocessing
import random


def knn_query(tree, d_query, k):
    start_q = timeit.default_timer()
    dist, ind = tree.query(d_query, k)
    end_q = timeit.default_timer() - start_q

    #  print(ind)  # indices of k closest neighbors
    # print("\n")
    #  print(dist)  # distances to k closest neighbors
    print("\n")

    return end_q


print("Which database do you want?\n")
print("1 for Earthquake Data\n")
print("2 for Santorini Lidar Data\n")
print("3 for Paris Static Scanner Data\n")
print("4 for Artificial Data\n")

choose_data = int(input("Enter a number from above: "))

N = input("How many points do you want to query for? Enter a number: ")
if N == "":
    print("We will use the existing points! ")

# num_neigh = int(input("How many nearest neighbors do you want? Enter a number: "))
num_neigh = 1000
data1, data2, data3, data4 = preprocessing.KD_tree()
if choose_data == 1:
    data = data1
    # get n random rows from data
    if N == "":
        query_data = data
    else:
        query_data = data[0:int(N)]
elif choose_data == 2:
    data = data2
    # get n random rows from data
    if N == "":
        query_data = data
    else:
        query_data = data[0:int(N)]
elif choose_data == 3:
    data = data3
    # get n random rows from data
    if N == "":
        query_data = data
    else:
        query_data = data[0:int(N)]
else:
    data = data4
    # get n random rows from data
    if N == "":
        query_data = data
    else:
        query_data = data[0:int(N)]
result = 0

start = timeit.default_timer()
kd_tree = KDTree(data)
end = timeit.default_timer() - start
print("KD-tree construction time is:", end)
print("\n")

for i in range(100):
    result += knn_query(kd_tree, query_data, num_neigh)
print("Average time for knn queries after 100 iterations: ", result / 100)

