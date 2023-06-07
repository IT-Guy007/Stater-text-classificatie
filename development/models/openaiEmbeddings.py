import pandas as pd
import time
import openai
import json
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Setup openAI API key
openai.api_key = '<api_key>'

# OpenAI model for embedding complaints: text-embedding-ada-002
embedding_model = 'text-embedding-ada-002'

# Retrieve the data from the database
input_datapath = pd.read_csv('StaterData.csv')

# Limit test size due to performance issues
data = input_datapath.loc[:200]

# Create and return embedding
def get_embedding(text, model='text-embedding-ada-002'):
    # sleep 1 second to prevent reaching rate limit. Limit: 60 requests per min.
    time.sleep(1)
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

data['Embedding'] = data['Clean consumer complaint'].apply(lambda x: get_embedding(x))
data.to_csv("StaterDataEmbeddings.csv", index=False)

# Prepare the feature matrix X and target vector y
X = data['Embedding'].apply(lambda x: json.loads(x)).tolist()
y = data['Issue']

# Reshape the embeddings into a 2D array
X = np.array(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# Create an instance of the random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=200, random_state=2)

# Train the random forest classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)