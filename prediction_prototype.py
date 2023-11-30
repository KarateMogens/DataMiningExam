# %%
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_dfs
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, multilabel_confusion_matrix
from pprint import pprint

data_df, data_ordinal_df, holdout_df = get_dfs()
# %%
# TRYING TO PREDICT TRACK POPULARITY BASED ON MUSICAL FEATURES
X = data_ordinal_df
y = data_df['track_popularity']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly_features', PolynomialFeatures()),
    ('regression', Ridge())
])
param_grid = {
    'poly_features__degree': [1, 2, 3,],
    'regression__alpha': [0.1, 0.5, 1, 10,]
}

grid_search = GridSearchCV(pipe, param_grid, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
# Fuldstændig elendig performance
pprint(metrics)
print(grid_search.best_params_)

# %%
# TRYING TO PREDICT GENRE BASED ON MUSICAL FEATURES
X = data_ordinal_df
y = data_df['playlist_genre']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_test, y_test)
y_pred = dt_model.predict(X_test)
# overraskende god performance
# multilabel_confusion_matrix(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

# sns.heatmap(conf_matrix, cmap='Blues')
conf_matrix_df = pd.DataFrame(
    conf_matrix, index=np.unique(y), columns=np.unique(y))

# heatmap for confusion matrix
# used chatGPT to make it pretty
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_df, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('confusion matrix')
plt.show()
# plotting the feature importance extracted from the dtree model
feature_importances = dt_model.feature_importances_
feature_names = X.columns
importances = pd.DataFrame(
    feature_importances, feature_names, columns=['importance'])
plt.figure(figsize=(10, 8))
importances.plot(kind='bar', legend=False)
plt.title('feature importances')
plt.xlabel('features')
plt.ylabel('importance')
plt.show()

# K-MEANS TEST

# %%
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
