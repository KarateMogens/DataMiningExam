#%%
import pandas as pd
import numpy as np
import random
from utils import get_dfs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import csv

#%%
CLUSTERS = 25

path = "song_recommendation.csv"
if not os.path.exists(path=path):

    print("No .csv file detected. Clustering and creating file for recommendation")
    #Build csv data for recommendation in case .csv is not present
    data_df, audio_features_df, holdout_df = get_dfs()
    audio_features_df = audio_features_df.drop(columns=["mode", "key", "loudness", "duration_ms", "track_popularity"])

    transformer = StandardScaler()
    scaled_audio_features = transformer.fit_transform(audio_features_df)
    k_means_model = KMeans(init='k-means++', n_clusters=CLUSTERS, random_state=0).fit(scaled_audio_features)
    data_df['cluster'] = k_means_model.labels_

    print(data_df.head())
    data_df.to_csv(path_or_buf=path, index=False)

data_df = pd.read_csv("song_recommendation.csv")


# %%

while True:
    print("Please supply a valid track-id:")
    track_id = input()
    song = data_df.loc[(data_df['track_id'] == track_id)]
    if song.empty:
        print('Song not found. Please try again')
        continue
    #print(song.iloc[0])
    print(f"Listening to {song['track_name'].iloc[0]} by artist {song['track_artist'].iloc[0]}")
    cluster_songs = data_df.loc[(data_df['cluster'] == song['cluster'].iloc[0]) & (data_df['track_popularity'].ge(70)) & (data_df['track_id'] != song['track_id'].iloc[0])]
    print(cluster_songs.head(20))
    recommended_song = cluster_songs.sample()
    #print(recommended_song.iloc[0])
    print(f"Recommended song is {recommended_song['track_name'].iloc[0]} by artist {recommended_song['track_artist'].iloc[0]}")
   

# %%
