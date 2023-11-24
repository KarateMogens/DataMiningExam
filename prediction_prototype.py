# %%
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

# %%
