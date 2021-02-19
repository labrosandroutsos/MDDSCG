import timeit
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt


# Για βάση καββαδία με σεισμούς
# read flash.dat to a list of lists
# data = [i.strip().split() for i in open("./gshap_pub/GSHPUB.DAT").readlines()]

# write it as a new CSV file
# with open("./dataset.csv", "w") as f:
#    writer = csv.writer(f)
#    writer.writerows(data)


def knn_query(d, d_query, k):
    start = timeit.default_timer()
    kd_tree = KDTree(d)
    end = timeit.default_timer() - start
    print("KD-tree construction time is:", end)
    print("\n")

    start_q = timeit.default_timer()
    dist, ind = kd_tree.query(d_query, k)
    end_q = timeit.default_timer() - start_q

  #  print(ind)  # indices of k closest neighbors
   # print("\n")
  #  print(dist)  # distances to k closest neighbors
    print("\n")

    return end_q


print("Which database do you want?\n")
print("0 for Large Seismic Data\n")
print("1 for Test Seismic Data\n")
print("2 for Lidar Scan Data\n")  # den yparxei akoma

choose_data = int(input("Enter a number from above: "))

N = input("How many points do you want to query for? Enter a number: ")
if N == "":
    print("We will use the existing points! ")

num_neigh = int(input("How many nearest neighbors do you want? Enter a number: "))
print("\n")
if choose_data == 0:

    # Βάση Καββαδία
    column_names = ['Latitude', 'Longitude', 'PGA']
    X_data = pd.read_csv('dataset.csv', delimiter=',', names=column_names, header=None)
    X = X_data.dropna()
    X = X.reset_index(drop=True)
    X_array = X.to_numpy()
    data = X_array
    # get n random rows from data
    if N == "":
        query_data = data
    else:
        query_data = data[np.random.choice(int(N), size=int(N), replace=False)]
else:  # choose_data == 1:

    # Βάση σεισμών Λάμπρου
    Y = pd.read_csv("database.csv")
    Y = Y[['Latitude', 'Longitude', 'Magnitude']]
    Y_array = Y.to_numpy()
    data = Y_array
    # get n random rows from data
    if N == "":
        query_data = data
    else:
        query_data = data[np.random.choice(int(N), size=int(N), replace=False)]

# else: (για choose_data==2)
# den exw akoma data

result = 0
for i in range(100):
    result += knn_query(data, query_data, num_neigh)
print("Average time for knn queries after 100 iterations: ", result / 100)
