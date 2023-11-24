
# %%
# IMPORTS:
import pandas as pd
import seaborn as sns

# %%
data_df = pd.read_csv("spotify_songs.csv")

print(list(data_df.keys()))

# Ordinal datapoints: 'track_popularity', 'track_album_release_date', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'

# Checking for extreme values/malformed data entries.

nominal_columns = ['track_popularity', 'track_name', 'track_id', 'track_artist', 'playlist_subgenre', 'track_album_id',
                   'playlist_genre', 'track_album_name', 'playlist_name', 'playlist_id', 'playlist_genre', 'track_album_release_date']
ordinal_columns = [x for x in list(data_df.keys()) if x not in nominal_columns]

data_ordinal_df = data_df.drop(['track_popularity', 'track_name', 'track_id', 'track_artist', 'playlist_subgenre', 'track_album_id',
                               'playlist_genre', 'track_album_name', 'playlist_name', 'playlist_id', 'playlist_genre', 'track_album_release_date'], axis=1)

data_df.describe()


# %%
# Full plot of 2-way data combinations (and distribution for each attribute)
sns.pairplot(data_df)

# %%

correlation_matrix = data_df.corr()

# To-do: Plot as confusion-matrix with gradient from invisible to green (0 to 1)
correlation_matrix

# %%
# Questions/Hypotheses:
# -

# Brainstorm:
# Supervised Learning:
# - Decision tree / Naïve bayes til at definere genre/playlist
# Unsupervised Learning:
# - Clustering for at definere nye genre-grupperinger
# -
# Correlation Analysis:
# - Hvilke features har størst betydning for
# Feature reduction:
# - PCA
