# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# stopwords
nltk.download('stopwords')
nltk.download('punkt')

# word normalizer
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# constants
conn = sqlite3.connect('StaterData.db')
query = "SELECT * FROM 'mortgage complaints'"
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

# remove stopwords and return the text
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return " ".join([token for token in nltk.word_tokenize(text) if token.lower() not in stop_words])

def clean_text(text):
    # remove numerical values
    text = re.sub(r'\d+', '', text)

    # remove punctuation marks
    text = re.sub(r'[^\w\s]', '', text)

    # remove links and URLs
    text = re.sub(r'http\S+', '', text)

    # remove leading/trailing white space and convert to lowercase
    text = text.strip().lower()

    return text

# lemmitize the word based on its part of speech (POS) tag
def lemmatize_word(word, tag, lemmatizer):
    # map POS tag to WordNet POS tag
    if tag.startswith('J'):
        # adjective
        wn_tag = 'a'
    elif tag.startswith('V'):
        # verb
        wn_tag = 'v'
    elif tag.startswith('N'):
        # noun
        wn_tag = 'n'
    elif tag.startswith('R'):
        # adverb
        wn_tag = 'r'
    else:
        wn_tag = None

    # lemmatize the word
    if wn_tag:
        lemma = lemmatizer.lemmatize(word, wn_tag)
    else:
        lemma = word

    return lemma

# lemmatize the sentence that is already tokenized
def lemmatize_sentence(lemmatizer, tokens):
    # part-of-speech (POS) tag each word
    pos_tags = nltk.pos_tag(tokens)

    # lemmatize each word based on its POS tag
    lemmas = [lemmatize_word(word, tag, lemmatizer) for word, tag in pos_tags]

    # join the lemmas back into a sentence
    lemmatized_sentence = ' '.join(lemmas)

    return lemmatized_sentence

def remove_non_alphabetica_char_and_x(text):
    # remove non alphabetical characters
    alphabetical_text = [re.sub('[^a-zA-Z]+', '', word) for word in text]

    # remove x from the text with regex
    alphabetical_text = [word for word in alphabetical_text if not re.match('^x+$', word)]
    return ' '.join(alphabetical_text)

def clean_complaint(complaint):
        lemmatizer = WordNetLemmatizer()
        cleaned_complaint = remove_stopwords(complaint)
        cleaned_complaint = clean_text(cleaned_complaint)
        # tokenize the sentence into words
        tokens = nltk.word_tokenize(cleaned_complaint)
        cleaned_complaint = lemmatize_sentence(lemmatizer, tokens)
        tokenized_complaint = word_tokenize(cleaned_complaint)
        cleaned_complaint = remove_non_alphabetica_char_and_x(tokenized_complaint)
        return cleaned_complaint

# load data and clean complaint narrative
print("load data")
data = pd.read_sql_query(query, conn, dtype=dtypes)
df = data[['Consumer complaint narrative', 'Issue']].copy()

print("clean text")
df['Clean consumer complaint'] = df['Consumer complaint narrative'].copy()
df['Clean consumer complaint'] = df['Clean consumer complaint'].apply(lambda x: clean_complaint(x))

# data = pd.read_csv('cleaned_complaints.csv', dtype=dtypes)
# print(data.head())
# print(data.info())
# df = data[['Clean consumer complaint', 'Issue']].copy()
# print("data loaded")
# print(df.shape)


# get the independent and dependent variables as x and y
x = df['Clean consumer complaint']
y = df['Issue']
# get all categories of y for the results
cat_issue = y.cat.categories

print("splitting data into train and test sets")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

print("vectorize text")
vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b[a-zA-Z]+\b')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# multinomial logistic regression
print("creating model")
logregression = LogisticRegression(n_jobs=4, solver='saga', multi_class='multinomial', max_iter=1000, random_state=2)

print("fitting model")
logregression.fit(X_train_tfidf, y_train)

print("predicting test data")
y_pred = logregression.predict(X_test_tfidf)

# print scores
print(f"score:{logregression.score(X_test_tfidf, y_test)}")
print(f"accuracy:{accuracy_score(y_test, y_pred)}")
print(f"f1 score:{f1_score(y_test, y_pred, average='weighted')}")

# cm = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=cat_issue, index=cat_issue)
# cm = (cm.transpose()/cm.sum(axis=1)).transpose()

# sns.heatmap(cm, annot=True)
# plt.show()

# print report
print(classification_report(y_test, y_pred, target_names=cat_issue))

# test the model with custom text
print("testing model with custom text")
new_complaint = "My mortgage was sold to Roundpoint company in XX/XX/2023. I got a letter in XXXX from my prior lender, XXXX XXXX stating this was to occur and the next payment to be made would go to Roundpoint. But this is false. The information in the letter is not true. I think it is a mistake of yours"
new_complaint_vectorized = vectorizer.transform([new_complaint])
issue_pred = logregression.predict(new_complaint_vectorized)
print('Predicted issue:', issue_pred[0])

# save model with joblib
print("saving model")
joblib.dump(logregression, 'logregression_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

# print("creating gridsearch")
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],
#     'solver': ['sag', 'saga'],
#     'max_iter': [100, 1000, 2500, 5000],
#     'penalty': ['l1', 'l2', 'elasticnet', None]
# }
# model = LogisticRegression(multi_class='multinomial', n_jobs=4, random_state=2)
# clf = GridSearchCV(model, param_grid, verbose=3)
# clf.fit(X_train_tfidf, y_train)
# print(clf.best_params_)
# print(clf.best_estimator_)
# print(clf.best_score_)