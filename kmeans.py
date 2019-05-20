import numpy as np
import matplotlib.pyplot as plt

from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# plt.figure(figsize=(6, 6))

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=1500, centers=3, n_features=4, random_state=700)

y_pred = KMeans(n_clusters=5, random_state=random_state).fit_predict(X)
#y_pred = mixture.GaussianMixture(n_components = 6, covariance_type='spherical').fit_predict(X)


plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Number of Blobs")

plt.show()
