# Imports
import pandas as pd
import sqlalchemy

# Defining constants
url = 'https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?date_received_max=2023-04-24&date_received_min=2011-12-01&field=all&format=csv&lens=product&no_aggs=true&product=Mortgage&size=375533&sub_lens=sub_product&trend_depth=5&trend_interval=month'
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

# Downloading dataset and
print("Starting download")
DS1_data = pd.read_csv(url, low_memory = False, dtype = dtypes, parse_dates = parse_dates)
print("Download complete")

# Cleaning
print("Cleaning Data")
DS1_data[['Timely response?','Consumer disputed?']] = DS1_data[['Consumer disputed?','Timely response?']].replace({'Yes': True, 'No':False}).astype(bool)
DS1_data['Consumer consent provided?'] = DS1_data['Consumer consent provided?'].replace({'Consent provided': True, '':False}).astype(bool)
DS1_data = DS1_data[pd.notnull(DS1_data['Consumer complaint narrative'])]   #Drops the missing rows with no  complaint.
DS1_data.drop(columns=['Sub-issue'])
print("Data cleaned")

# Exporting
db = sqlalchemy.create_engine('sqlite:///StaterData.db')
print("Exporting to database")
DS1_data.to_sql("mortgage complaints",db,if_exists="replace")
print("Done!")

