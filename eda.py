
# %%
# IMPORTS:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_dfs

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
