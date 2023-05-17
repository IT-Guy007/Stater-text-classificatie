print("Importing libraries...")
#Importing the libraries
import pandas as pd
import ssl
import datetime

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Exporting model
import joblib

# Stop words
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Word normalizer
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Cleaning unneeded repetitive words
import re

# Setting ssl settings for MacOS
ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
ssl_context.load_default_certs()

# get the date of yesterday
yesterday = datetime.date.today() - datetime.timedelta(days=1)
yesterday_str = yesterday.strftime('%Y-%m-%d')
date_formats = ['%Y-%m-%d', '%m/%d/%Y']
url = f'https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?date_received_max={yesterday_str}&date_received_min=2011-12-01&field=all&format=csv&lens=product&no_aggs=true&product=Mortgage&size=375533&sub_lens=sub_product&trend_depth=5&trend_interval=month'
parse_dates = ['Date received', 'Date sent to company']
dtypes = {'Date received': str,
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
          'Complaint ID':int}

# Downloading dataset
print("Downloading data...")
data = pd.read_csv(url, low_memory = False, dtype = dtypes, parse_dates = parse_dates)

# Cleaning
print("Cleaning data...")
data[['Timely response?','Consumer disputed?']] = data[['Consumer disputed?','Timely response?']].replace({'Yes': True, 'No':False}).astype(bool)
data['Consumer consent provided?'] = data['Consumer consent provided?'].replace({'Consent provided': True, '':False}).astype(bool)
data[data['Consumer complaint narrative'].notna()]
data['Consumer complaint narrative'] = data['Consumer complaint narrative'].str.lower()
data = data.drop(columns=['Sub-issue'])


# Remove stopwords
print("Removing stop words...")
stop_words = set(stopwords.words('english'))

remove_stopwords = lambda text: " ".join([token for token in nltk.word_tokenize(text) if token.lower() not in stop_words])

data['Consumer complaint narrative'] = data['Consumer complaint narrative'].apply(remove_stopwords)

# Removing hashtags, urls, commas, etc.
print("Cleaning text...")
def clean_text(text):
    # Remove numerical values
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation marks
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove links and URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove leading/trailing white space and convert to lowercase
    text = text.strip().lower()
    
    return text

data['Consumer complaint narrative'] = data['Consumer complaint narrative'].apply(clean_text)



# Normalizing words, (play, playing, played) -> play
print("Normalizing words...")
# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function to lemmatize a single word based on its part of speech (POS) tag
def lemmatize_word(word, tag):
    # Map POS tag to WordNet POS tag
    if tag.startswith('J'):
        # Adjective
        wn_tag = 'a'
    elif tag.startswith('V'):
        # Verb
        wn_tag = 'v'
    elif tag.startswith('N'):
        # Noun
        wn_tag = 'n'
    elif tag.startswith('R'):
        # Adverb
        wn_tag = 'r'
    else:
        wn_tag = None
    
    # Lemmatize the word
    if wn_tag:
        lemma = lemmatizer.lemmatize(word, wn_tag)
    else:
        lemma = word
    
    return lemma

# Define a function to lemmatize a sentence
def lemmatize_sentence(sentence):
    # Tokenize the sentence into words
    tokens = nltk.word_tokenize(sentence)
    
    # Part-of-speech (POS) tag each word
    pos_tags = nltk.pos_tag(tokens)
    
    # Lemmatize each word based on its POS tag
    lemmas = [lemmatize_word(word, tag) for word, tag in pos_tags]
    
    # Join the lemmas back into a sentence
    lemmatized_sentence = ' '.join(lemmas)
    
    return lemmatized_sentence


data['Consumer complaint narrative'] = data['Consumer complaint narrative'].apply(lemmatize_sentence)


# Tokenization, creating into sets of words
print("Tokenizing words...")
data['Consumer complaint narrative']= [word_tokenize(entry) for entry in data['Consumer complaint narrative']]

# Remove only tokenized words that are not alphabetic or only x
print("Removing non-alphabetic words...")
data['Consumer complaint narrative'] = data['Consumer complaint narrative'].apply(lambda x: [re.sub('[^a-zA-Z]+', '', word) for word in x])
data['Consumer complaint narrative'] = data['Consumer complaint narrative'].apply(lambda x: [word for word in x if not re.match('^x+$', word)])


# Split the data into training and testing sets.
print("Splitting data...")
train_data, test_data, train_labels, test_labels = train_test_split(data['Consumer complaint narrative'], data['Issue'], test_size=0.1)

# Convert back to strings
train_data = [' '.join(tokens) for tokens in train_data]
test_data = [' '.join(tokens) for tokens in test_data]

# Create a TF-IDF vectorizer.
vectorizer = TfidfVectorizer(
    stop_words="english",
    token_pattern=r'\b[a-zA-Z]+\b',
    analyzer="word",
    use_idf=True,
    smooth_idf=True,
    norm=None,
    tokenizer=None,
    preprocessor=None
)
# Fit the vectorizer to the training data.
print("Fitting vectorizer...")
vectorizer.fit(train_data)

# Create a TF-IDF vectorizer object with unigrams, bigrams, and trigrams as features
vectorizer = TfidfVectorizer(ngram_range=(1, 3))

# Create a support vector machine classifier.
clf = SVC(kernel='linear')

# Fit the vectorizer on the training data
train_tfidf_vectors = vectorizer.fit_transform(train_data)

# Transform the testing data using the fitted vectorizer
test_tfidf_vectors = vectorizer.transform(test_data)

# Create a support vector machine classifier.
clf = SVC(kernel='linear')

# Train the classifier on the TF-IDF vectors.
print("Training classifier...")
clf.fit(train_tfidf_vectors, train_labels)

# Predict the labels of the testing data
pred_labels = clf.predict(test_tfidf_vectors)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(test_labels, pred_labels)
count = data.shape[0] // 1000

# Export the model to a file.
print("Exporting model...")
filename = f"model({count}K, {accuracy:.1%}).joblib"
joblib.dump(clf, filename)

print(f"Exported to: {filename}")
