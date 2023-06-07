import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

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
df = data[['Clean consumer complaint', 'Issue']].copy()

# Split the data into training and testing sets: x = complaint and y = issue
X_train, X_test, y_train, y_test = train_test_split(df['Clean consumer complaint'], df['Issue'], test_size=0.3, random_state=2)

# Vectorize the data with tfidf
vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b[a-zA-Z]+\b')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create multinomial logistic regression
logregression = LogisticRegression(C=1, multi_class='multinomial', n_jobs=4, random_state=2, solver='saga')

# Fit and predict
logregression.fit(X_train_tfidf, y_train)
y_pred = logregression.predict(X_test_tfidf)

# function to classify a complaint
def ask_question(text):
    question_tfidf = vectorizer.transform([text])
    prediction = logregression.predict(question_tfidf)
    return prediction[0]