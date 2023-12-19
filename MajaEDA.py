#%%
import pandas as pd
import numpy as np

df = pd.read_csv("spotify_songs.csv")

#%%
df.head(1)

#%%
# Below is generated from a former project on the same data set.
# It is a great way to get a initial feel of the data set.

# Returns df only with artists with more than 100 tracks on the chart
artist_counts = df.track_artist.value_counts()
top_100_artists = artist_counts[artist_counts >= 100].index.tolist()
top_artists = df[df['track_artist'].isin(top_100_artists)]

#%%
# Assigning top artists a popularity score based on number of tracks on chart
top_artists_popularity_score = top_artists['track_artist'].value_counts().to_dict()
top_artists['popularity'] = top_artists['track_artist'].map(top_artists_popularity_score)

#%%
# Normalizing popularity score
X_min = 20
X_max = 161

top_artists['normalized_score'] = (top_artists['popularity'] - X_min) / (X_max - X_min)
top_artists

#%%
# Plotting top artists energy and danceability 

import matplotlib.pyplot as plt

top_artists.plot(kind='scatter', x='danceability', y='energy', alpha=0.4, s=top_artists['popularity']/100, label='popularity', figsize=(15,10), c='normalized_score', cmap = plt.get_cmap('jet'), colorbar=True)
plt.legend()

#%%
# Below code is generated after gaining an initial impression of the data set above

df_float = pd.DataFrame(df, columns=['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness'])
df_object_playlist = pd.DataFrame(df, columns=['playlist_name', 'playlist_id', 'playlist_genre', 'playlist_subgenre'])
df_object_track = pd.DataFrame(df, columns=['track_id', 'track_name', 'track_artist', 'track_album_id', 'track_album_name', 'track_album_release_date'])

df_object_track.track_album_release_date.value_counts

#%%
corr_matrix = df_float.corr()
print(corr_matrix)

#%%
corr_matrix['danceability'].sort_values(ascending=True)

#%%
# Correlation heatmap
import seaborn as sns

plt.figure(figsize=(14,14))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

#%%
sns.pairplot(df_float, corner=True)
plt.show()

#%%
df_float.plot(kind='scatter', x='energy', y='loudness', alpha=0.01)
plt.show()

#%%
df_float.plot(kind='scatter', x='acousticness', y='energy', alpha=0.01)
plt.show()

#%%
df_float.plot(kind='scatter', x='danceability', y='speechiness', alpha=0.01)
plt.show()