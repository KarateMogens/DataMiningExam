
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


# %%
# plotter alle musical features ift genre
for feature in data_ordinal_df.columns:
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='playlist_genre', y=feature, data=data_df)
    plt.show()


# %%
plt.hist(data_df['year'], bins=62)
# %%
sns.barplot(data_df, x='playlist_genre', y='track_popularity')
# %%
sns.barplot(data_df, x='playlist_subgenre', y='track_popularity')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
for feature in data_ordinal_df.keys():
    sns.barplot(data_df, x='playlist_subgenre', y=feature)
    plt.xticks(rotation=45, ha='right')
    plt.show()
# %%
for feature in data_ordinal_df.keys():
    sns.barplot(data_df, x='playlist_genre', y=feature)
    plt.xticks(rotation=45, ha='right')
    plt.show()

# %%
plot_features_by_year(data_df)
# %%
# MÅSKE ER DE NÆSTE TO LIDT OVERKILL
plot_features_by_year_compare_genres(data_df)
# %%
plot_yearly_features_by_genre(data_df)


# INVESTIGATING MOST EXTREME TRACKS FOR EACH MUSICAL FEATURE
# %%
for feature in data_ordinal_df.keys():
    feature_max = data_df.loc[data_df[feature].idxmax()]
    feature_min = data_df.loc[data_df[feature].idxmin()]
    print(f"\n############{feature}#########\n ")
    print(f"min {feature}: \n{feature_min}")
    print(f"\nfeature: {feature}\ntrack: {feature_max['track_name']}\n")
    print(f"max {feature}: \n{feature_max}")
    print(f"\nfeature: {feature}\ntrack: {feature_min['track_name']} \n ")


# EXTRA JUST FOR
# %%
# bare for at bevise at 'key' og 'mode' ikke nødvendigvis er korrekte
bjork = data_df[data_df['track_artist'] == 'Björk']
bjork.head()
# %%
# HUMBLE. er både kategoriseret som pop og rap
s = data_df[data_df['track_artist'] == 'Kendrick Lamar']
s.head()
# # s['key'].head()
# # %%
# s = data_df[data_df['track_artist'] == 'Taylor Swift']
# s.head()


# # %%
# sns.histplot(data=data_df, x='year', hue='playlist_genre', multiple='stack')

# # %%
# # ved ikke helt med de her. Prøver bare at se hvor stor en andel hvert år er fra hver genre
# sns.histplot(data=data_df, x='year', hue='playlist_genre',
#              multiple='stack', stat='proportion', cumulative=True)

# # %%
# sns.relplot(data_df.sample(500), x='energy',
#             y='danceability', size='track_popularity', hue='key')


# %%
