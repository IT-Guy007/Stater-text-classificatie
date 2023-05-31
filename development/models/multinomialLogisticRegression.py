import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

csv_file = '../StaterData.csv'
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
logregression = LogisticRegression(n_jobs=4, solver='saga', multi_class='multinomial', max_iter=1000, random_state=2)

# Fit and predict
logregression.fit(X_train_tfidf, y_train)
y_pred = logregression.predict(X_test_tfidf)

# Print scores
print(f"score:{logregression.score(X_test_tfidf, y_test)}")
print(f"accuracy:{accuracy_score(y_test, y_pred)}")
print(f"f1 score:{f1_score(y_test, y_pred, average='weighted')}")

cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=cat_issue, index=cat_issue)
cm = (cm.transpose()/cm.sum(axis=1)).transpose()

sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_test, y_pred, target_names=cat_issue))

# Test the model with custom text
new_complaint = "My mortgage was sold to Roundpoint company in XX/XX/2023. I got a letter in XXXX from my prior lender, XXXX XXXX stating this was to occur and the next payment to be made would go to Roundpoint. But this is false. The information in the letter is not true. I think it is a mistake of yours"
new_complaint_vectorized = vectorizer.transform([new_complaint])
issue_pred = logregression.predict(new_complaint_vectorized)
print('Predicted issue:', issue_pred[0])

print("creating gridsearch")
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['sag', 'saga'],
    'max_iter': [100, 1000, 2500, 5000],
    'penalty': ['l1', 'l2', 'elasticnet', None]
}

# Subselect the first 10000 samples for the GridSearch
X_train_tfidf_subselect = X_train_tfidf[:10000]
y_train_subselect = y_train[:10000]

model = LogisticRegression(multi_class='multinomial', n_jobs=4, random_state=2)
grid = GridSearchCV(model, param_grid, verbose=3)
grid.fit(X_train_tfidf_subselect, y_train_subselect)
print(grid.best_params_)
print(grid.best_estimator_)
print(grid.best_score_)