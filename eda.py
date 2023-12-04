
# %%
# IMPORTS:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_dfs
from sklearn.cluster import KMeans

# define functions for plotting features by year


def plot_features_by_year(df):
    for feature in data_ordinal_df.columns:
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=df, x='year', y=feature)
        plt.title(f"{feature} over years")
        plt.show()


def plot_yearly_features_by_genre(df):
    for genre in df['playlist_genre'].unique():
        plot_features_by_year(data_df[data_df['playlist_genre'] == genre])


def plot_features_by_year_compare_genres(df):
    for feature in data_ordinal_df.columns:
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=data_df, x='year', y=feature, hue='playlist_genre')
        plt.title(f"{feature} over years")
        plt.show()


data_df, data_ordinal_df, holdout_df = get_dfs()
data_df.describe()

# %%
# Full plot of 2-way data combinations (and distribution for each attribute)
sns.pairplot(data_df)

# %%
# TAKES A LONG TIME!
sns.pairplot(data_df, hue='playlist_genre')
# %%
# ALSO SLOW
sns.pairplot(data_df, hue='mode')
# %%
correlation_matrix = data_ordinal_df.corr()

# To-do: Plot as confusion-matrix with gradient from invisible to green (0 to 1)
correlation_matrix
sns.heatmap(correlation_matrix, cmap='Blues')

# # %%
# rock = data_df[data_df['playlist_genre'] == 'rock']
# # %%
# sns.pairplot(rock)


# %%
# plotter alle musical features ift genre
for feature in data_ordinal_df.columns:
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='playlist_genre', y=feature, data=data_df)
    plt.show()


# %%

data_df.columns
# %%
# create 'year' columns to be able to plot feature by year
data_df['track_album_release_date'] = pd.to_datetime(
    data_df['track_album_release_date'], format='ISO8601')
data_df['year'] = data_df['track_album_release_date'].dt.year
# %%
plot_features_by_year(data_df)
# %%
# MÅSKE ER DE NÆSTE TO LIDT OVERKILL
plot_features_by_year_compare_genres(data_df)
# %%
plot_yearly_features_by_genre(data_df)

# %%
# %%
# K-MEANS
# sum of squared distances
ssd = []

for k in range(2, 15):
    model = KMeans(n_clusters=k)
    model.fit(X)
    # add the models sum of squared distances (inertia_) to ssd list to be able to do elbow plot
    ssd.append(model.inertia_)

# plot ssd
plt.plot(range(2, 15), ssd, 'o--')
plt.xlabel("k value")
plt.ylabel("sum of squared distances")
# %%
# data_df['playlist_genre'].unique()
# k=6 virker som et meget godt bud, især siden det er det antal genre der allerede er defineret
model = KMeans(n_clusters=6)
data_df['cluster'] = model.fit_predict(X)
data_df.head()

sns.countplot(data_df, x=data_df['cluster'])
# %%
sns.countplot(data_df, x=data_df['playlist_genre'])
# %%
sns.countplot(data=data_df, x='playlist_subgenre')
plt.xticks(rotation=45, ha='right')
plt.show()
# %%

# %%
sns.countplot(data_df, x='playlist_genre', hue='cluster')
# %%
cross_tab = pd.crosstab(data_df['playlist_genre'], data_df['cluster'])
sns.heatmap(cross_tab, cmap='Blues')
# %%
cross_tab
# %%
# EDA
# find ud af hvad der kendetegner hver genre og cluster
feature_stats = {}
for feature in X.keys():
    genre_stats = {}
    cluster_stats = {}
    for genre in data_df['playlist_genre'].unique():
        genre_stats[genre] = data_df[feature][data_df['playlist_genre']
                                              == genre].mean()

    for cluster in data_df['cluster'].unique():
        cluster_stats[cluster] = data_df[feature][data_df['cluster']
                                                  == cluster].mean()
    feature_stats[f'{feature}_genre'] = genre_stats
    feature_stats[f'{feature}_cluster'] = cluster_stats

# %%
for feature_key, stats in feature_stats.items():
    if 'genre' in feature_key:
        plt.bar(stats.keys(), stats.values())
        plt.title(f'mean values for {feature_key}')
        plt.xlabel('genre')
        plt.ylabel(feature_key)
        plt.show()

# %%
for feature_key, stats in feature_stats.items():
    if 'cluster' in feature_key:
        plt.bar(stats.keys(), stats.values())
        plt.title(f'mean values for {feature_key}')
        plt.xlabel('cluster')
        plt.ylabel(feature_key)
        plt.show()


# %%
# bare for at bevise at 'key' og 'mode' ikke nødvendigvis er korrekte
bjork = data_df[data_df['track_artist'] == 'Björk']
bjork.head()
# %%
# HUMBLE. er både kategoriseret som pop og rap
s = data_df[data_df['track_artist'] == 'Kendrick Lamar']
s.head()
# s['key'].head()
# %%
s = data_df[data_df['track_artist'] == 'Taylor Swift']
s.head()

# %%
# det midst dansable nummer (typisk latin regn...)
mindance = data_df.loc[data_df['danceability'].idxmin()]
mindance
# %%
# vanvittigt nummer
maxdance = data_df.loc[data_df['danceability'].idxmax()]
maxdance
# %%
# igen latin? energiske bølgelyde til at sove
maxenergy = data_df.loc[data_df['energy'].idxmax()]
maxenergy
# %%
minenergy = data_df.loc[data_df['energy'].idxmin()]
minenergy
# %%
maxdur = data_df.loc[data_df['duration_ms'].idxmax()]
maxdur
# %%
maxloud = data_df.loc[data_df['loudness'].idxmax()]
maxloud

# %%
# %%
maxpopular = data_df.loc[data_df['track_popularity'].idxmax()]
maxpopular
# %%
# kan ikke finde den
minpopular = data_df.loc[data_df['track_popularity'].idxmin()]
minpopular
# %%
# tempo giver kun mening fordi de ikke kan holde tempoet
maxlive = data_df.loc[data_df['liveness'].idxmax()]
maxlive

# %%
minval = data_df.loc[data_df['valence'].idxmax()]
minval

# %%
maxsp = data_df.loc[data_df['speechiness'].idxmax()]
maxsp
# %%

plt.hist(data_df['year'])
# %%
sns.histplot(data=data_df, x='year', hue='playlist_genre', multiple='stack')

# %%
# ved ikke helt med de her. Prøver bare at se hvor stor en andel hvert år er fra hver genre
sns.histplot(data=data_df, x='year', hue='playlist_genre',
             multiple='stack', stat='proportion', cumulative=True)

# %%
sns.relplot(data_df.sample(500), x='energy',
            y='danceability', size='track_popularity', hue='key')


# %%
