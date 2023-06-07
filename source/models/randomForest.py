# Various imports used for notebook
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

# Read cleaned dataset created from complaintCleaner.py
data = pd.read_csv('StaterData.csv')

# Define your features and target
X = data['Clean consumer complaint']  # text_column is the name of the column in your dataset that contains the text data
y = data['Issue']  # target_column is the name of the column in your dataset that contains the target variable

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# Define params from GridSearch which performed the best
param_grid = {
    'n_estimators': 200
}

# Vectorize your text data using TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Create a Random Forest model
rf_model = RandomForestClassifier(random_state=2)

# Set the params for the model
rf_model.set_params(**param_grid)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Predict the target variable for the testing data
y_pred = rf_model.predict(X_test)

def text_to_prediction_random_forest(text):
    custom_text_bow = vectorizer.transform([text])
    predicted_issue = rf_model.predict(custom_text_bow)

    return predicted_issue

# Define your features and target
X = data['Consumer complaint narrative']  # text_column is the name of the column in your dataset that contains the text data
y = data['Issue']  # target_column is the name of the column in your dataset that contains the target variable

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# Vectorize your text data using a bag-of-words approach
vectorizer = TfidfVectorizer(stop_words=['english'])
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Create a Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, random_state=2)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Get feature importances
importances = rf_model.feature_importances_

# Create a DataFrame with feature names and importances
feature_importances = pd.DataFrame({'feature': vectorizer.get_feature_names(), 'importance': importances})

# Sort the DataFrame by importance
feature_importances = feature_importances.sort_values('importance', ascending=False)

top_features = feature_importances.head(10)['feature'].values.tolist()

# Define your features and target
X = data['Consumer complaint narrative']  # text_column is the name of the column in your dataset that contains the text data
y = data['Issue']  # target_column is the name of the column in your dataset that contains the target variable

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# Vectorize your text data using a bag-of-words approach
vectorizer = TfidfVectorizer(stop_words='english', vocabulary=top_features)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Create a Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, random_state=2)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Predict the target variable for the testing data
y_pred = rf_model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred, average='macro')

# # Define the parameter grid to search over
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [10, 20, 30, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
# }
#
# # Create a Random Forest model
# rf_model = RandomForestClassifier(random_state=2)
#
# # Create a GridSearchCV object
# grid_search = GridSearchCV(
#     estimator=rf_model,
#     param_grid=param_grid,
#     cv=5,  # number of folds for cross-validation
#     scoring='accuracy',
#     # n_jobs=-1,  # use all available CPU cores
# )
#
# # Fit the grid search to the training data
# grid_search.fit(X_train, y_train)
# 
# # Print the best hyperparameters and the corresponding F1 score
# print('Best hyperparameters:', grid_search.best_params_)
# print('Best accuracy score:', grid_search.best_score_)