import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

csv_file = 'StaterData.csv'
dtypes = {
    'Date received': str,
    'Product': "category",
    'Sub-product': "category",
    'Issue': "category",
    'Sub-issue':"category",
    'Consumer complaint narrative':str,
    'Company public response':str,
    'Company':"category",
    'State':"category",
    'ZIP code':str,
    'Tags':"category",
    'Consumer consent provided?':str,
    'Submitted via':"category",
    'Date sent to company':str,
    'Company response to consumer':str,
    'Timely response?':str,
    'Consumer disputed?':str,
    'Complaint ID':int,
    'Clean consumer complaint':str,
}

# Read data from csv
data = pd.read_csv(csv_file, dtype=dtypes)
print(data.head())
print(data.info())
df = data[['Clean consumer complaint', 'Issue']].copy()

# Set the independent and dependent variables as x and y
x = df['Clean consumer complaint']
y = df['Issue']

# Get all categories of y for the results
cat_issue = y.cat.categories

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

# Vectorize the data with tfidf
vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b[a-zA-Z]+\b')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create multinomial logistic regression
logregression = LogisticRegression(C=1, multi_class='multinomial', n_jobs=4, random_state=2, solver='saga')

# Fit and predict
logregression.fit(X_train_tfidf, y_train)
y_pred = logregression.predict(X_test_tfidf)

print(classification_report(y_test, y_pred, target_names=cat_issue))

# Use GridSearch to find the best parameters for the model
# print("creating gridsearch")
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'solver': ['sag', 'saga'],
#     'max_iter': [100, 1000, 2500, 5000],
#     'penalty': ['l1', 'l2', 'elasticnet', None]
# }

# # Subselect the first 10000 samples for the GridSearch
# X_train_tfidf_subselect = X_train_tfidf[:10000]
# y_train_subselect = y_train[:10000]

# model = LogisticRegression(multi_class='multinomial', n_jobs=4, random_state=2)
# grid = GridSearchCV(model, param_grid, verbose=3)
# grid.fit(X_train_tfidf_subselect, y_train_subselect)
# print(grid.best_params_)
# print(grid.best_estimator_)
# print(grid.best_score_)