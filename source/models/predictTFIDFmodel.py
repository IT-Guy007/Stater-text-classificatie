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
data = pd.read_csv("../dataPreparation/TrainData.csv", low_memory=False, dtype=dtypes, parse_dates=parse_dates)

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
Vectorized_Data = vectorizer.fit_transform(data['Consumer complaint narrative'])

feature_names = vectorizer.get_feature_names_out()

# Defines the dictionary of english words
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

# Calculate the normalized count of issue categories
IssueCountNormalized = data['Issue'].value_counts(normalize=True)

# Get unique values from the "Issue" column
unique_issues = data['Issue'].unique()

# Concatenate the unique issues into a single string
all_issues = ' '.join(unique_issues)

# Split the concatenated string into individual words
all_words = all_issues.split()

# Get unique words
Mortgage_Terms = set(all_words)

# Function creates an array of every word in text and gives it an score
def tfidf_custom_scoring(input_text):
    # Fit the data to the vectorizer
    vectorizer.fit(data['Consumer complaint narrative'])

    # Adds the input from the user to the fit using an transform
    transformed_data = vectorizer.transform([input_text])

    # Get all the feature name(words instead of numbers) corresponding to the array
    feature_names = vectorizer.get_feature_names_out()

    # calculate initial scores and store them in a dictionary
    scores_dict = {name: score for name, score in zip(feature_names, transformed_data.toarray()[0])}

    # adjust scores based on term length and update the dictionary
    for name in feature_names:
        if name in Mortgage_Terms:
            scores_dict[name] *= (1 + 0.01*len(name) + 1.5)
        else:
            scores_dict[name] *= (1 + 0.01*len(name))

    # create a new array of adjusted scores in the same order as the feature names
    adjusted_scores = np.zeros(len(feature_names))
    for INDEX, term in enumerate(feature_names):
        adjusted_scores[INDEX] = scores_dict[term]

    # return a tuple of the feature names and adjusted scores
    return feature_names, adjusted_scores

def classifiyComplaintTFIDF(user_input):
    # Get an array of words with corresponding scores
    data = tfidf_custom_scoring(user_input)

    # Create a dictionary with feature names as keys and scores as values
    scores_dict = {name: score for name, score in zip(data[0], data[1])}

    # Create a new array of adjusted scores in the same order as the feature names
    adjusted_scores = np.zeros(len(feature_names))
    for i, term in enumerate(feature_names):
        adjusted_scores[i] = scores_dict[term]

    # Set the top 3 words to a list
    Top3Words = []

    def get_top3_words():
        # Get the index of the highest score in the scores array
        index_max = adjusted_scores.argmax()

        while index_max != 0 and len(Top3Words) <= 2:
            # Check if there are any vowels in the top word, if not, select a new word
            while True:
                vowels = {"a", "e", "i", "o", "u", "A", "E", "I", "O", "U"}
                # Get the corresponding word
                top_word = feature_names[index_max]
                if any(char in vowels for char in top_word) and top_word in english_vocab:
                    Top3Words.append(top_word)
                break

            # Get the corresponding word
            top_word = str(data[0][index_max])

            # If the word is in the English vocabulary, add it to the top 3 list
            if top_word in english_vocab:
                Top3Words.append(top_word)

            # Set the current score of this word to 0 to select the second most popular word
            adjusted_scores[index_max] = 0
            index_max = adjusted_scores.argmax()

    get_top3_words()

    # Check all past result classifications
    def check_corresponding_word(relevant_word):
        return data[(data["top_word"].str[0] == relevant_word[0]) & data["top_word"].str[1].isin(relevant_word)]

    filtered_df = check_corresponding_word(Top3Words)

    # Count the occurrences of each issue and get the most common one, as long as there are words left
    value_counts = filtered_df["Issue"].value_counts()
    
    # Concatenate the 'IssueCountNormalized' and 'value_counts' dataframes along the columns axis
    NormalizedTable = pd.concat([IssueCountNormalized, value_counts], axis=1, keys=('perc', 'valuecount'))

    # Calculate the 'Endscores' by dividing the 'valuecount' column by the 'perc' column in 'NormalizedTable'
    Endscores = NormalizedTable.valuecount / NormalizedTable.perc

    # Assign the calculated 'Endscores' as a new column in 'NormalizedTable'
    NormalizedTable["Endscores"] = Endscores

    # Create a new column in 'NormalizedTable' containing the keys from 'IssueCountNormalized' converted to a list
    NormalizedTable["IssueName"] = IssueCountNormalized.keys().tolist()

    # Locate the row in 'NormalizedTable' with the maximum value in the 'Endscores' column
    Toprow = NormalizedTable.loc[NormalizedTable['Endscores'].idxmax()]

    # Return the value in the 'IssueName' column of the row with the highest 'Endscores' value
    return Toprow.IssueName

print("Your question will be in the following category: "+classifiyComplaintTFIDF(input("What question do you want to categorize? ")))
