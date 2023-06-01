import json

from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer

from .tasks import get_response
#import requests
#from requests.adapters import HTTPAdapter
#from urllib3.util import Retry
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import nltk
nltk.download('words')
from googletrans import Translator, constants

import google.generativeai as palm
from google.api_core import retry
#from tqdm.auto import tqdm
#from keras import layers
#import sys
#import platform
#import sklearn as sk
#import tensorflow as tf
#import keras
#import re
#import tqdm


translator = Translator()

# Create a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

param_grid = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
}

# Set the params for the model
rf_model.set_params(**param_grid)

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
DS1_data = pd.read_csv("static/TrainDataEmbedded.csv", low_memory=False, dtype=dtypes, parse_dates=parse_dates)

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
#Vectorized_Data = vectorizer.fit_transform(DS1_data['Consumer complaint narrative'])
Vectorized_Data = vectorizer.fit_transform(DS1_data['Consumer complaint narrative'].values.astype('U'))

# Train the model on the training data


# === Bard modeling ===
#palm.configure(api_key='AIzaSyDl-GxBo7WsAQiw99q5yKACHkQ7c-ysIQ8')
#models = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods]

#model = models[0]

issue_to_int = {
    "Trouble during payment process": 0,
    "Struggling to pay mortgage": 1,
    "Loan servicing, payments, escrow account": 2,
    "Applying for a mortgage or refinancing an existing mortgage": 3,
    "Loan modification,collection,foreclosure": 4,
    "Closing on a mortgage": 5,
    "Application, originator, mortgage broker": 6,
    "Credit decision / Underwriting": 7,
    "Incorrect information on your report": 8,
    "Settlement process and costs": 9,
    "Problem with a credit reporting company's investigation into an existing problem": 10,
    "Improper use of your report": 11,
    "Credit monitoring or identity theft protection services": 12,
}

# Create a new column called `Issue_Int`
DS1_data["Issue_Int"] = DS1_data["Issue"].map(issue_to_int)


def make_embed_text_fn(model):
    def embed_fn(text: str) -> list[float]:
        # Using the palm model, generate the embeddings for the text
        return palm.generate_embeddings(model=model, text=text)['embedding']
    return embed_fn

# Creates the embedding of the given text
def create_embeddings(model, text):
    # Generate the embedding using the make_embed_text_fn function
    embedding = make_embed_text_fn(model)(text)
    # Return the embedding
    return embedding

# Creates the embedding for the train and the testdata

class ChatConsumer(WebsocketConsumer):
    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        get_response.delay(self.channel_name, text_data_json)

        async_to_sync(self.channel_layer.send)(
            self.channel_name,
            {
                "type": "chat_message",
                "text": {"msg": text_data_json["text"], "source": "user"},
            },
        )
        rf_model.fit(Vectorized_Data, DS1_data['Issue'])  #
        #rf_model.fit(Vectorized_Data, DS1_data['Embeddings'])  #

        translation = translator.translate(text_data_json["text"])
        score = vectorizer.transform([translation.text])
        #DS1_data['Consumer complaint narrative'].iloc[0] = translation.text
        #predictiondataset = create_embeddings(model, translation.text)
        #print(predictiondataset)
        #score = vectorizer.transform([predictiondataset])

        prediction = rf_model.predict(list(score.toarray()))
        #prediction = rf_model.predict(list((np.array(predictiondataset)).reshape(1, -1)))

        prediction_nl = translator.translate("Your question belongs to the: " + prediction[0] + " category", dest=translation.src)

        async_to_sync(self.channel_layer.send)(
            self.channel_name,
            {
                "type": "chat_message",
                "text": {"msg": prediction_nl.text, "source": "bot"},
            },
        )

    def chat_message(self, event):
        text = event["text"]
        self.send(text_data=json.dumps({"text": text}))
