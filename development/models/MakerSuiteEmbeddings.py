# Imports
import google.generativeai as palm

import re
import tqdm
import tensorflow as tf
import keras
import numpy as np
import sklearn as sk
import pandas as pd

from keras import layers
from sklearn.model_selection import train_test_split
from google.api_core import retry
from tqdm.auto import tqdm
import sys
import platform

# Print kernal stats
print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

# Select the right model
palm.configure(api_key='<>')
models = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods]
model = models[0]

# Import the data, and spit the data
data = pd.read_csv("StaterData.csv")
df_train, df_test = train_test_split(data, test_size=0.3, random_state=2)

# Python progress bar 
tqdm.pandas()

def make_embed_text_fn(model):
  @retry.Retry(timeout=300.0)
  def embed_fn(text: str) -> list[float]:
    # Using the palm model generate the embeddings for 
    return palm.generate_embeddings(model=model, text=text)['embedding']
  return embed_fn

# Creates the embedding of the send dataframe
def create_embeddings(model, data):
  # Adds the column embeddings with the corrosponding embedding
  data['Embeddings'] = data['Clean consumer complaint'].progress_apply(make_embed_text_fn(model))
  return data

# Creates the embedding for the train and the testdata
df_train = create_embeddings(model, df_train)
df_test = create_embeddings(model, df_test)

# Calling the API from Palm and convert the text to vectors
def build_classification_model(input_size: int, num_classes: int) -> keras.Model:
  inputs = x = keras.Input(input_size)
  x = layers.Dense(input_size, activation='relu')(x)
  x = layers.Dense(num_classes, activation='sigmoid')(x)
  return keras.Model(inputs=[inputs], outputs=x)

# Derive the embedding size from the first training element.
embedding_size = len(df_train['Embeddings'].iloc[0])

# Give your model a different name, as you have already used the variable name 'model'
classifier = build_classification_model(embedding_size, len(df_train['Clean consumer complaint'].unique()))
classifier.summary()

classifier.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   optimizer = keras.optimizers.Adam(learning_rate=0.001),
                   metrics=['accuracy'])

# Tensorflow model training

NUM_EPOCHS = 20
BATCH_SIZE = 8

# Split the x and y components of the train and validation subsets.
y_train = df_train['Issue_Int']
x_train = np.stack(df_train['Embeddings'])
y_val = df_test['Issue_Int']
x_val = np.stack(df_test['Embeddings'])

# Train the model for the desired number of epochs.
callback = keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)

history = classifier.fit(x=x_train,
                         y=y_train,
                         validation_data=(x_val, y_val),
                         callbacks=[callback],
                         batch_size=BATCH_SIZE,
                         epochs=NUM_EPOCHS,) 

classifier.evaluate(x=x_val, y=y_val, return_dict=True)
