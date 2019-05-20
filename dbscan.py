import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs


centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)


db = DBSCAN(eps=0.3, min_samples=10).fit(X)

labels = db.labels_

unique_labels = set(labels)
n_clusters_ = len(unique_labels) - 1

colors = cycle('bgrcmyk')

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    my_members = labels == k
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')


plt.title("Number of estimated clusters usin DBSCAN: %d" % (n_clusters_))
plt.show()