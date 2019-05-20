import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth


# centers = [[1, 1], [-1, -1], [1, -1]]
# X, _ = make_blobs(n_samples=10000, centers=3, cluster_std=0.5, random_state=1300)

centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)


bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

labels = ms.labels_
cluster_centers = ms.cluster_centers_

unique_labels = set(labels)
n_clusters_ = len(unique_labels)

plt.figure(1)
plt.clf()

colors = cycle('bgrcmyk')
for k, col in zip(unique_labels, colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=10)

plt.title("Number of estimated clusters usin MeanShift: %d" % n_clusters_)
plt.show()