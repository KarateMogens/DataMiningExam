
# %%
# IMPORTS:
import pandas as pd
import seaborn as sns
from utils import get_dfs

# %%
data_df, data_ordinal_df, holdout_df = get_dfs()
data_df.describe()

# %%
data_ordinal_df
# %%
# Full plot of 2-way data combinations (and distribution for each attribute)
sns.pairplot(data_df)

# %%
sns.pairplot(data_df, hue='playlist_genre')
# %%
sns.pairplot(data_df, hue='mode')
# %%
correlation_matrix = data_ordinal_df.corr()

# To-do: Plot as confusion-matrix with gradient from invisible to green (0 to 1)
correlation_matrix
sns.heatmap(correlation_matrix, cmap='Blues')

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
