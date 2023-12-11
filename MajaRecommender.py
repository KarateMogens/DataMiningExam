#%%
from utils import get_dfs
(full_df, float_df, holdout) = get_dfs()
full_df.columns

# %% #imports and data loading
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv("spotify_songs.csv")


# %%
# First experiment: Setting range myself via dictionary by which I determine mood

# selecting relevant columns
# i pick the column 'energy' since it tells me how energetic the song is
# i pick the column 'valence' because in the data set this is an indicator of positivity of the song
# i pick the column 'danceability' since i might want to use it for a category later
columns = ["track_name", "energy", "valence", "danceability"]

# new datafram with selected columns
data = full_df.loc[:, columns]
data

# %%
# i scale up the columns valence and energy to create better room for mood application
#df['valence_multiplied'] = df['valence'] * 10
#df['energy_multiplied'] = df['energy'] * 10

# defining moods and their ranges
# i decided to specify a significant/unique range to each item/mood

#%% # mood dict
moods = {
    'energy boost': {'valence': (0.9, 1.0), 'energy': (0.9, 1.0)},
    'i am happy': {'valence': (0.6, 0.8), 'energy': (0.6, 0.8)},
    'just give me a loving background soundscape': {'valence': (0.5, 0.5), 'energy': (0.5, 0.5)},
    'i am sad and a bit depressed': {'valence': (0.0, 0.4), 'energy': (0.0, 0.4)}
}

# %% # defining a function that assigns moods as labels defined by a specific range
def assign_mood(row):
    for mood, ranges in moods.items():
        if all(ranges[feature][0] <= row[feature] <= ranges[feature][1] for feature in ['valence', 'energy']):
            print(f"Assigned '{mood}' to row {row.name}")
            return mood
    print(f"Assigned 'Other' to row {row.name}: valence={row['valence']}, energy={row['energy']}")
    return 'Other'

# %% # appending the new column with the moods
data['mood_label'] = data.apply(assign_mood, axis=1)

# %%
# checking the distribution of moods
data['mood_label'].value_counts()

# First experiment conclusion:
# Defining clusters myself didn't work

# %%

# Second experiment: Iterate a grid search look alike setup to find best combination of features
# that gives the highest possible silhouette score

from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# features to combine
columns_float = ["energy", "valence", "danceability", "loudness", "tempo"]

# defining new df with the chosen features
df_float1 = data.loc[:, columns_float]

# defining what my features are
features = columns_float

# create combinations of features up until four units in the combination
all_possible_feature_combinations = list(combinations(features, 4))

# convert combinations to a list of tuples
all_possible_feature_combinations = [tuple(combination) for combination in all_possible_feature_combinations]

# initialize variables to store the best feature combination and silhouette score
best_features = None
best_silhouette_score = -1  # Initialize with a low value

# standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_float1)

# specify the range of clusters to consider
cluster_range = range(2, 10)

# iterate over all possible feature combinations
for feature_combination in all_possible_feature_combinations:
    print(f"\nEvaluating Feature Combination: {feature_combination}")
    
    # Subsetting the DataFrame with selected features
    subset_data = df_float1[list(feature_combination)].values

    # Iterate over different numbers of clusters
    for num_clusters in cluster_range:
        print(f"\n  Evaluating Number of Clusters: {num_clusters}")

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(subset_data)

        # Evaluate clustering quality using silhouette score
        silhouette_avg = silhouette_score(subset_data, labels)
        print(f"    Silhouette Score: {silhouette_avg}")

        # Update if a better combination is found
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_features = feature_combination
            best_num_clusters = num_clusters

print("\nBest Feature Combination:", best_features)
print("Best Number of Clusters:", best_num_clusters)
print("Best Silhouette Score:", best_silhouette_score)

# Second experiment conclusion: T
# he best feature combination is all four features with a number of three clusters
# It gives a score at 0.62

# TODO: Apply labels to the clusters and build the recommender

# %%
