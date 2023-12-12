# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def get_dfs():
    df = pd.read_csv("spotify_songs.csv")
    # create holdout_df to save for final testing of models
    data_df, holdout_df = train_test_split(df, random_state=42, test_size=0.2)
    # Ordinal datapoints: 'track_popularity', 'track_album_release_date', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'

    nominal_columns = ['track_popularity', 'track_name', 'track_id', 'track_artist', 'playlist_subgenre', 'track_album_id',
                       'playlist_genre', 'track_album_name', 'playlist_name', 'playlist_id', 'playlist_genre', 'track_album_release_date']
    ordinal_columns = [x for x in list(
        data_df.keys()) if x not in nominal_columns]

    data_ordinal_df = data_df.drop(nominal_columns, axis=1)

    return data_df, data_ordinal_df, holdout_df

# %%
