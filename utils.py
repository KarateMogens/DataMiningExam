# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_dfs():
    data_df = pd.read_csv("spotify_songs.csv")

    # Ordinal datapoints: 'track_popularity', 'track_album_release_date', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'

    # Checking for extreme values/malformed data entries.

    nominal_columns = ['track_popularity', 'track_name', 'track_id', 'track_artist', 'playlist_subgenre', 'track_album_id',
                       'playlist_genre', 'track_album_name', 'playlist_name', 'playlist_id', 'playlist_genre', 'track_album_release_date']
    ordinal_columns = [x for x in list(
        data_df.keys()) if x not in nominal_columns]

    data_ordinal_df = data_df.drop(nominal_columns, axis=1)

    return data_df, data_ordinal_df
