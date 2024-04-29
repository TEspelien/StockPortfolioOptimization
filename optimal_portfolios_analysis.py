import pandas as pd
import numpy as np
import math

from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.metrics import silhouette_score

from matplotlib import pyplot as plt

p1_minima = pd.read_csv('p1_minima.csv')

# Create and fit the OPTICS model
optics_model = OPTICS(min_samples = 4, xi=0.007, min_cluster_size = 3)
optics_model.fit(p1_minima)

# Extract the cluster labels using DBSCAN
labels = optics_model.labels_

silhouette_avg = silhouette_score(p1_minima, labels)

print(labels)



