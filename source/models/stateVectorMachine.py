#imports
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# load data
data = pd.read_csv("StaterData.csv")

# split data
train_data, test_data, train_labels, test_labels = train_test_split(data['Clean consumer complaint'], data['Issue'], test_size=0.1, random_state=2)

# vectorize data
vectorizer = TfidfVectorizer(
    stop_words="english",
    token_pattern=r'\b[a-zA-Z]+\b',
    analyzer="word",
    use_idf=True
)

# Vectorize data
vectorizer.fit(train_data)

# If needed, use grid search to find best parameters
# clf = SVC(kernel='linear')

# parameters = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': ['scale', 'auto'],
#     'class_weight': ['balanced', None],
# }

# grid_search = GridSearchCV(clf, parameters, cv=5)
# grid_search.fit(train_tfidf_vectors, train_labels)

# print("Best parameters:", grid_search.best_params_)
# print("Best score:", grid_search.best_score_)

# Set model parameters
clf = SVC(C=10, class_weight='balanced', gamma='scale', kernel='linear')

# Fit model
train_tfidf_vectors = vectorizer.fit_transform(train_data)

# Transform test data
test_tfidf_vectors = vectorizer.transform(test_data)

# Fit model
clf.fit(train_tfidf_vectors, train_labels)

# Predict
pred_labels = clf.predict(test_tfidf_vectors)

# Evaluate
accuracy = accuracy_score(test_labels, pred_labels, normalize=True)
count = data.shape[0] // 1000

# Print the results
print(f"Accuracy: {accuracy * 100:.2f}% with({count}k samples)")

# Print classification report
predictions = clf.predict(test_tfidf_vectors)
report = classification_report(test_labels, predictions, zero_division=1)
print(report)
def text_to_prediction_svm(text):
    custom_text_bow = vectorizer.transform([text])
    predicted_issue = clf.predict(custom_text_bow)

    return predicted_issue