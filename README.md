# Text Classifier

This code provides a text classification model using a TF-IDF (Term Frequency-Inverse Document Frequency) approach. The model is trained on a dataset of consumer complaints and predicts the category of a given input text based on its content.

## Prerequisites

Make sure you have the following dependencies installed:

- pandas
- numpy
- scikit-learn (sklearn)
- nltk

You also need to download the English words corpus from NLTK by running the following command:

```python
import nltk
nltk.download('words')```

## Installation
Clone the repository or download the code files.
Install the required dependencies mentioned in the Prerequisites section.

## Usage
Prepare your training data in a CSV format with the following columns: 'Date received', 'Product', 'Sub-product', 'Issue', 'Sub-issue', 'Consumer complaint narrative', 'Company public response', 'Company', 'State', 'ZIP code', 'Tags', 'Consumer consent provided?', 'Submitted via', 'Date sent to company', 'Company response to consumer', 'Timely response?', 'Consumer disputed?', 'Complaint ID'.
Replace the file path in the line DS1_data = pd.read_csv("TrainData.csv", low_memory=False, dtype=dtypes, parse_dates=parse_dates) with the path to your training data file.
Run the code.

##Performance
The model achieved an accuracy of approximately 69% on a test dataset of 1000 samples. The results are printed as follows:

Correct predictions: [CorrectPrognosed]%.
Incorrect predictions: [WrongPrognosed]%.