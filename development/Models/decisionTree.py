# Imports
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


# Read csv, StaterData
df = pd.read_csv('/Users/laurensheberle/Github/StaterData.csv')

# Define your features and target, with x as text_column which contains the complaint in the dataset. And target_column which contains the target variable
X = df['Clean consumer complaint']  
y = df['Issue']                        

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# Vectorize your text data using a bag-of-words approach
vectorizer = TfidfVectorizer(stop_words='english',token_pattern=r'\b[a-zA-Z]+\b')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

feature_names = vectorizer.get_feature_names_out()
class_names = df['Issue']

# Gridsearch Define the parameter grid
best_params = {
    'criterion': 'gini',
    'max_depth': 14,
    'min_samples_split': 2,
    'min_samples_leaf': 1
    }

# Create a Decision tree model
model = DecisionTreeClassifier(random_state=2, **best_params)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the target variable for the testing data
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)