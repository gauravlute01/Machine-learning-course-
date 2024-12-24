# Simple k-means cluster
import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(0)

dim = 2
N = 1000
num_cluster = 4
iteration = 3

x = np.random.randn(N, dim)
y = np.random.randint(0, num_cluster, N)

mean = np.zeros((num_cluster, dim))
for t in range(iteration):
    for k in range(num_cluster):
        mean[k] = np.mean(x[y==k], axis=0)
    for i in range(N):
        dist = np.sum((mean - x[i])**2, axis = 1)
        pred = np.argmin(dist)
        y[i] = pred
for k in range(num_cluster):
    plt.scatter(x[y==k,0], x[y==k, 1])
plt.title('K-Means Clustering')
plt.show()
