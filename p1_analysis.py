import pandas as pd
import numpy as np
import math

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

p1_minima = pd.read_csv('p1_minima.csv')

# Example list of 20 5-dimensional vectors
vectors = np.random.rand(20, 5)

# Initialize variables
best_score = -1
best_n_clusters = -1

# Try different numbers of clusters
for n_clusters in range(3, 20):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(p1_minima)
    silhouette_avg = silhouette_score(p1_minima, cluster_labels)
    
    # Check if current number of clusters gives better silhouette score
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_n_clusters = n_clusters

# Use the best number of clusters
kmeans = KMeans(n_clusters=best_n_clusters, random_state=0).fit(p1_minima)
cluster_labels = kmeans.labels_

# Print cluster labels and best number of clusters
print("Best number of clusters:", best_n_clusters)
print("Cluster labels:", cluster_labels)
