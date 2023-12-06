#%%
# ------ Imports -----
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import random
from utils import get_dfs

#%%
# ----- Initialize Data -----

data_df, data_ordinal_df, holdout_df = get_dfs()
np.shape(data_ordinal_df)
print(data_ordinal_df.columns)

# Drop mode, key?
data_ordinal_df.describe()  
data_ordinal_df.drop(columns=["mode", "key"])

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
    print(sample_indices)

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

    #We have a lot of p_distances = 0.0, meaning we have duplicate data
    print(sum(np.where(a = 0, p_distances)))
    
    # print(q_distances)

    # Calculate actual Hopkins measure. 0.5 is uniform distribution (no inherent clustering), 1.0 is highly clustered
    H = sum(q_distances)/(sum(p_distances) + sum(q_distances))

    return H

# %%
print("Hopkins score:", hopkins_statistic(data_ordinal_df))

# %%
