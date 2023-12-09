#%%
# ------ Imports -----
import pandas as pd
import numpy as np
import random
from utils import get_dfs

# - Data handling -
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# - Hopkins Stats - 
from sklearn.neighbors import NearestNeighbors

# - Plotting -
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

# - K-means -
from sklearn.cluster import k_means

# - GMM -
from sklearn.mixture import GaussianMixture

#%%
# ----- Initialize Data -----

data_df, data_ordinal_df, holdout_df = get_dfs()

# Drop mode, key?
data_ordinal_df.describe()  
data_ordinal_df = data_ordinal_df.drop(columns=["mode", "key", "loudness", "track_popularity", "duration_ms", "tempo"])
data_ordinal_df.describe()  
#%%
def hopkins_statistic(X):
    # Normalize all data to the [0,1] range
    transformer = MinMaxScaler()
    X = transformer.fit_transform(X)

    # Fix variables
    rows, columns = X.shape[0], X.shape[1]
    sample_size = int(0.1*rows)

    # Determine samples (10% of data)
    sample_indices = random.sample(range(rows), sample_size)

    # Init nearest neighbor search object
    nearest_neighbors = NearestNeighbors(n_neighbors=1).fit(X)

    # Calculate nearest neighbor distances for q in Q (uniform random samples in X-space) and p in Q (samples from X)
    p_distances = []
    q_distances = []
    for i in range(sample_size):
        p = X[sample_indices[i]]
        q = np.random.uniform(low=0, high=1, size=columns)
        # Pick second neighbor for P, because p is its own nearest neighbor
        p_neighbor_dist = nearest_neighbors.kneighbors(X=[p], n_neighbors=2, return_distance=True)[0][0,1]
        q_neighbor_dist = nearest_neighbors.kneighbors(X=[q], n_neighbors=1, return_distance=True)[0][0,0]

        p_distances.append(p_neighbor_dist)
        q_distances.append(q_neighbor_dist)

    # Calculate actual Hopkins measure. 0.5 is uniform distribution (no inherent clustering), 1.0 is highly clustered
    H = sum(q_distances)/(sum(p_distances) + sum(q_distances))

    return H

# %%
print("Hopkins score:", hopkins_statistic(data_ordinal_df))
print("Not bad, not great..")

# %%
# ------            Method for plotting clustering + silhouette chart             -------

def plot_cluster(X, labels, nclusters):

    plt.clf()

    avg_score = silhouette_score(X=X, labels=labels)

    silhouette_scores = silhouette_samples(X=X, labels=labels)

    # 2 rows, 1 col plot
    superplot, (plot1, plot2) = plt.subplots(1, 2)
    superplot.set_figheight(7)
    superplot.set_figwidth(16)
    
    y_lower = 100
    for i in range(nclusters):

        # Get silhoutte values for each sample belonging to cluster i
        cluster_i_samples = silhouette_scores[labels == i]
        cluster_i_samples.sort()
        type(cluster_i_samples)
        n_samples = cluster_i_samples.shape[0]

        # Upper y-axis limit for plot
        y_upper = y_lower + n_samples

        # Fill y coordinates in the y-range and add scores of cluster i to plot
        cluster_color = plt.cm.rainbow(float(i/nclusters))
        y_coordinates = np.arange(y_lower, y_upper)
        plot2.fill_betweenx(y=y_coordinates, x1=0, x2=cluster_i_samples, color=cluster_color)
        
        # Updates y-range + space between
        y_lower = y_upper + 100

    # Plot the vertical line, showing the avg. silhouette score
    plot2.axvline(x=avg_score, color="red")
    plot2.set_yticks([])
    plot2.set_ylabel("Clusterings")
    plot2.set_xlim([-0.3,1.0])
    plot2.set_xlabel("Silhouette coefficients. Red line = avg.")
    plot2.set_title("Silhouette chart of clustering")
    

    # Transform data into 2-d for plottable represenation with PCA
    if X.shape[1] > 2:
        pca_model = PCA(n_components=2)
        X = pca_model.fit_transform(X)
    
    # Plot clustering
    plot1.set_title(f"Clustering of {nclusters} clusters in the 2-component PCA-space")
    plot1.scatter(X[:,0], X[:,1], c=labels, cmap='rainbow')
    plt.show()


#%%
# ------             K-means Model              -------

mean_squared_distances = []

# Scale data, expecting normal distribution
transformer = StandardScaler()
scaled_ordinal_data = transformer.fit_transform(data_ordinal_df)

for i in range(2, 10):

    # Create Kmeans-model with i clusters
    k_means_model = KMeans(n_clusters=i, n_init=5)
    k_means_model.fit(scaled_ordinal_data)
    
    # Get label for each track
    clustering_labels = k_means_model.labels_

    plot_cluster(X = scaled_ordinal_data, labels=clustering_labels, nclusters=i)
    
    # Add meansquaredistance for elbow plot
    mean_squared_distances.append([i, k_means_model.inertia_**(1/2)])

mean_squared_distances = np.array(mean_squared_distances)

# Plot elbow-chart
plt.plot(mean_squared_distances[:,0], mean_squared_distances[:,1], marker='o', linestyle='dashed')




# %%
# ------             Guassian Mixture-Model using EM              -------

# Scale data, expecting normal distribution
# transformer = StandardScaler()
# scaled_ordinal_data = transformer.fit_transform(data_ordinal_df)

# for i in range(2, 10):

#     # Create Kmeans-model with i clusters
#     guassian_model = GaussianMixture(n_components=i)
#     guassian_model.fit(scaled_ordinal_data)
    
#     # Get label for each track
#     clustering_labels = k_means_model.labels_

#     plot_cluster(X = scaled_ordinal_data, labels=clustering_labels, nclusters=i)
    
#     # Add meansquaredistance for elbow plot
#     mean_squared_distances.append([i, k_means_model.inertia_**(1/2)])
# %%
