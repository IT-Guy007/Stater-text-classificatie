# imports
import pandas as pd
import sqlite3

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

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
    'Clean consumer complaint':str
}

df = pd.read_sql_query(query, conn, dtype=dtypes)
print("data loaded")
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

print("clean text")
df = data.copy()
df['Clean consumer complaint'] = df['Clean consumer complaint'].apply(lambda x: clean_complaint(x))

print("save data in csv")
df.to_csv('cleanedComplaints.csv', index=False)