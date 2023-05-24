# imports
import math
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
    'Clean consumer complaint':str,
    'Consumer complaint cleaned?':bool
}

df = pd.read_sql_query(query, conn, dtype=dtypes)
print("data loaded")

# remove stopwords and return the text
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return " ".join([token for token in nltk.word_tokenize(text) if token.lower() not in stop_words])

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

# lemmitize the word based on its part of speech (POS) tag
def lemmatize_word(word, tag, lemmatizer):
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

# lemmatize the sentence that is already tokenized
def lemmatize_sentence(lemmatizer, tokens):
    # Part-of-speech (POS) tag each word
    pos_tags = nltk.pos_tag(tokens)

    # Lemmatize each word based on its POS tag
    lemmas = [lemmatize_word(word, tag, lemmatizer) for word, tag in pos_tags]

    # Join the lemmas back into a sentence
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
        # Tokenize the sentence into words
        tokens = nltk.word_tokenize(cleaned_complaint)
        cleaned_complaint = lemmatize_sentence(lemmatizer, tokens)
        tokenized_complaint = word_tokenize(cleaned_complaint)
        cleaned_complaint = remove_non_alphabetica_char_and_x(tokenized_complaint)
        return cleaned_complaint

# one for loop in batches to, remove stopwords, clean text, lemmatize, tokenize, and remove x or non alphabetic characters
def clean_complaints(df_column):
    cleaned_complaints_df = pd.DataFrame(columns=[df_column.name])
    batch_size = 10000
    batch_index = 0
    number_of_batches = math.ceil(len(df_column) / batch_size)
    while batch_index < number_of_batches:
        print(f"batch {batch_index+1} of {number_of_batches}")
        cleaned_complaints = []
        complaints = df_column[batch_index * batch_size: (batch_index + 1) * batch_size]
        for complaint in complaints:
            complaint = clean_complaint(complaint)
            cleaned_complaints.append(complaint)
        clean_complaints_df = pd.DataFrame(cleaned_complaints, columns=[df_column.name])
        cleaned_complaints_df = pd.concat([cleaned_complaints_df, clean_complaints_df], axis=0)
        batch_index += 1
        del complaints
        del complaint
        del cleaned_complaints
        del clean_complaints_df
    return cleaned_complaints_df[df_column.name]

print("cleaning text")
df['Clean consumer complaint'] = clean_complaints(df['Consumer complaint narrative']).reset_index(drop=True)

df.to_csv('cleaned_complaints.csv', index=False)

# print("saving cleaned data to sqlite database")
# def update_table(df, conn):
#     batch_size = 10000
#     batch_index = 0
#     number_of_batches = math.ceil(len(df) / batch_size)
#     while batch_index < number_of_batches:
#         print(f"batch {batch_index+1} of {number_of_batches}")
#         rows = df[batch_index * batch_size: (batch_index + 1) * batch_size]
#         for index, row in rows.iterrows():
#             # clean_complaint = row['Clean consumer complaint']
#             complaint = clean_complaint(row['Consumer complaint narrative'])
#             complaint_id = row['Complaint ID']
#             query = """
#                         UPDATE 'mortgage complaints'
#                         SET 'Clean consumer complaint' = ?, 'Consumer complaint cleaned?' = ?
#                         WHERE 'Complaint ID' = ?;
#                     """
#             values = (complaint, 1, complaint_id)
#             conn.execute(query, values)
#             conn.commit()
#         batch_index += 1
#
# update_table(df, conn)