# %%
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_dfs, print_confusion_matrix, plot_feature_importances, year_to_decade
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, multilabel_confusion_matrix
from pprint import pprint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

data_df, data_ordinal_df, holdout_df = get_dfs()


def classification_metrics(X, y, model, y_test, y_pred):
    classification_report(y_test, y_pred)

    print_confusion_matrix(y_test, y_pred, y)
    plot_feature_importances(model, X)


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
# dropping 'year' since it is not a musical feature (also dropping 'mode' and 'key' since they do not make any difference in the result)
X = data_ordinal_df.drop(['year', 'key', 'mode'], axis=1)
y = data_df['playlist_genre']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# overraskende god performance
classification_metrics(X, y, model, y_test, y_pred)


# PREDICT DECADE
# %%
df = year_to_decade(data_df)
y = df['decade']

X = data_ordinal_df.drop(['year'], axis=1)
# Apparently very little information is needed for classifying songs based on year
# X = X[['speechiness', 'danceability']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

classification_metrics(X, y, model, y_test, y_pred)


# %%
# PREDICTING PlAYLIST GENRE BASED ON track_name AND track_album_name


data_df['text'] = data_df['track_name'] + ' ' + data_df['track_album_name']
X = data_df['text']
y = data_df['playlist_genre']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


pipe = Pipeline([
    # ('count', CountVectorizer()),

    ('tfidf', TfidfVectorizer(ngram_range=(1, 6))),
    ('naive_bayes', MultinomialNB()),
    # ('svc', SVC()),
    # ('lr', LogisticRegression()),
    # ('dt', DecisionTreeClassifier()),
    # ('rf', RandomForestClassifier()),
    # ('knn', KNeighborsClassifier(n_neighbors=300)),
    # ('gb', GradientBoostingClassifier()),
    # ('mlp', MLPClassifier())
])

pipe.fit(X_train, y_train)


ypred = pipe.predict(X_test)
print(classification_report(y_test, ypred))
# ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test)
print_confusion_matrix(y_test, ypred, y)
# %%

# COMBINING TEXT AND MUSICAL FEATURES FOR GENRE PREDICTION


X = data_ordinal_df
X['text'] = data_df['track_name'] + ' ' + data_df['track_album_name']
y = data_df['playlist_genre']
# preprocessing steps for different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(ngram_range=(1, 6)), 'text'),
        # using numeric columns only
        ('scaler', StandardScaler(), X.select_dtypes(include='number').columns)
    ]
)
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('lr', LogisticRegression())
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


pipe.fit(X_train, y_train)
ypred = pipe.predict(X_test)
print(classification_report(y_test, ypred))
# ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test)
print_confusion_matrix(y_test, ypred, y)
# %%


# PREDICTING SUBGENRE BASED ON MUSICAL FEATURES
X = data_ordinal_df.drop(['year', 'key', 'mode'], axis=1)
y = data_df['playlist_subgenre']
X = X.drop('text', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# overraskende god performance
# multilabel_confusion_matrix(y_test, y_pred)
classification_metrics(X, y, model, y_test, y_pred)

# %%
# PREDICTING SUBGENRE BY COMBINED TEXT AND MUSICAL FEATURES
X = data_ordinal_df
X['text'] = data_df['track_name'] + ' ' + data_df['track_album_name']
y = data_df['playlist_subgenre']
# preprocessing steps for different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(ngram_range=(1, 6)), 'text'),
        # using numeric columns only
        ('scaler', StandardScaler(), X.select_dtypes(include='number').columns)
    ]
)
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('lr', LogisticRegression())
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


pipe.fit(X_train, y_train)
ypred = pipe.predict(X_test)
print(classification_report(y_test, ypred))
# ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test)
print_confusion_matrix(y_test, ypred, y)
# %%
