# %%
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, multilabel_confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from utils import get_dfs, plot_feature_importances, print_confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# %% 

data_df, data_ordinal_df, holdout_df = get_dfs()

print(data_ordinal_df)

# %%

########################################################################################
########################################################################################

########################    Random Forest Generator   #####################################

########################################################################################
########################################################################################
rhythm_related_columns = ['tempo', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence']

#cols for preproc
preproc = data_ordinal_df.drop('danceability', axis=1)
#X = data_ordinal_df[rhythm_related_columns]
X = data_ordinal_df.drop('danceability', axis=1)
y = data_ordinal_df['danceability']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)


min_max = MinMaxScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ('mm', min_max, list(preproc.columns)),
    ])

rf = RandomForestRegressor(n_estimators=200, random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf', rf)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Squared Error: " + str(mse))
print("R2: " + str (r2))
print("Mean Absolute Error: " + str (mae))

########################################################################################
########################################################################################

########################    GRADIENT BOOSTING REGRESSOR    #####################################

########################################################################################
########################################################################################

# %%

preproc = data_ordinal_df.drop('danceability', axis=1)

X = data_ordinal_df.drop('danceability', axis=1)
y = data_ordinal_df['danceability']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

min_max = MinMaxScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('mm', min_max, list(preproc.columns)),
    ])

gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=8, random_state=42)

pl = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', gbr)
])

pl.fit(X_train, y_train)

y_pred = pl.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Squared Error: " + str(mse))
print("R2: " + str (r2))
print("Mean Absolute Error: " + str (mae))

# %%
