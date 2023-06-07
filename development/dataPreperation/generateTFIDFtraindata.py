import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('words')

# Define the data types for each column
dtypes = {
    'Date received': str,
    'Product': "category",
    'Sub-product': "category",
    'Issue': "category",
    'Sub-issue': "category",
    'Consumer complaint narrative': str,
    'Company public response': str,
    'Company': "category",
    'State': "category",
    'ZIP code': str,
    'Tags': "category",
    'Consumer consent provided?': str,
    'Submitted via': "category",
    'Date sent to company': str,
    'Company response to consumer': str,
    'Timely response?': str,
    'Consumer disputed?': str,
    'Complaint ID': int
}

# Define the columns to parse as dates
parse_dates = ['Product', 'Date received', 'Date sent to company']

# Read the CSV file with specified data types and parse dates
DS1_data = pd.read_csv("Data/complaints-2023-04-25_05_07.csv", low_memory=False, dtype=dtypes, parse_dates=parse_dates)

# Convert 'Timely response?' and 'Consumer disputed?' columns to boolean values
DS1_data[['Timely response?', 'Consumer disputed?']] = DS1_data[['Timely response?', 'Consumer disputed?']].replace({'Yes': True, 'No': False}).astype(bool)

# Convert 'Consumer consent provided?' column to boolean values
DS1_data['Consumer consent provided?'] = DS1_data['Consumer consent provided?'].replace({'Consent provided': True, '': False}).astype(bool)

# Drop rows with missing complaint narratives
DS1_data.dropna(subset=['Consumer complaint narrative'], inplace=True)

# Drop the 'Sub-issue' column as it is not needed
DS1_data.drop(columns=['Sub-issue'], inplace=True)

# Replace alle X occurences with emty strings, to avoid it from being the most important word.
DS1_data['Consumer complaint narrative'] = DS1_data['Consumer complaint narrative'].str.replace('X', '')

# Calculate the normalized count of issue categories
IssueCountNormalized = DS1_data['Issue'].value_counts(normalize=True)

# Create a TfidfVectorizer with optimized settings
vectorizer = TfidfVectorizer(stop_words='english',              # Exclude common English words
                             token_pattern=r'\b[a-zA-Z]+\b',    # Consider only alphabetic tokens
                             analyzer='word',                   # Analyze at the word level
                             use_idf=True,                      # Apply inverse document frequency weighting
                             smooth_idf=True,                   # Apply smoothing to idf weights
                             strip_accents='ascii',
                             min_df=2,
                             norm='l2')

# Fit and transform the vectorizer to get an score per word in an array returned
Vectorized_Data = vectorizer.fit_transform(DS1_data['Consumer complaint narrative'])

# Add the scores array back into the dataframe.
DS1_data['TF-IDF scores'] = list(Vectorized_Data.toarray())

# Defines the dictionary of english words
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

# Gets the list of words from the vectorizer
feature_names = vectorizer.get_feature_names_out()

# Get unique values from the "Issue" column
unique_issues = DS1_data['Issue'].unique()

# Concatenate the unique issues into a single string
all_issues = ' '.join(unique_issues)

# Split the concatenated string into individual words
all_words = all_issues.split()

# Get unique words
Mortgage_Terms = set(all_words)

# Set an empty top 3 words list.
top_words = []


for i in range(Vectorized_Data.shape[0]):
    print(i)
    # Get the array with scores from this row
    row_scores = Vectorized_Data[i].toarray()[0]

    # Generate a dictionary with each word and then the Scores
    scores_dict = {name: score for name, score in zip(feature_names,row_scores)}

    # Loop over all the words and adjust the score
    for name in feature_names:
        # If term is in categoryname, add to score
        if name in Mortgage_Terms:
            scores_dict[name] *= (1 + 0.02*len(name) + 0.2)
        else:
            scores_dict[name] *= (1 + 0.02*len(name))

    # Create a new array of adjusted scores in the same order as the feature names
    adjusted_scores = np.zeros(len(feature_names))
    for i, term in enumerate(feature_names):
        adjusted_scores[i] = scores_dict[term]

    #Set the top 3 to a empty list.
    Top3Words = []

    # Get the index of the highest scoring word
    max_index = adjusted_scores.argmax()

    # Iterate until 3 top words found or list of words empty
    while max_index != 0 and len(Top3Words) <= 2:
        #Check if there are any vowels in the topword, if not select new word
        while True:
            vowels = {"a", "e", "i", "o", "u", "A", "E", "I", "O", "U"}
            # Get the corresponding word
            top_word = feature_names[max_index]
            if any(char in vowels for char in feature_names[max_index]) and top_word in english_vocab:
                Top3Words.append(top_word)
            break

        #Sets the current score of this word to 0 to select the second most popular word
        adjusted_scores[max_index] = 0
        max_index = adjusted_scores.argmax()
    # Voeg het bijbehorende woord toe aan de lijst van top_words
    top_words.append(Top3Words)

# Add the list of topwords to the dataframe
DS1_data['top_word'] = top_words

# Save the generated table to a new csv Dataset
DS1_data.to_csv('TrainData.csv')