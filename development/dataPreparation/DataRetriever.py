import pandas as pd
import datetime
import sqlalchemy
# get the date of yesterday
yesterday = datetime.date.today() - datetime.timedelta(days=1)
yesterday_str = yesterday.strftime('%Y-%m-%d')

# create the URL with the date of yesterday
url = f'https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?date_received_max={yesterday_str}&date_received_min=2011-12-01&field=all&format=csv&lens=product&no_aggs=true&product=Mortgage&size=375533&sub_lens=sub_product&trend_depth=5&trend_interval=month'

parse_dates = ['Date received', 'Date sent to company']
dtypes = {'Date received': str,
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
          'Complaint ID': int}

DS1_data = pd.read_csv(url, low_memory=False, dtype=dtypes, parse_dates=parse_dates)
DS1_data[['Timely response?', 'Consumer disputed?']] = DS1_data[['Consumer disputed?', 'Timely response?']].replace(
    {'Yes': True, 'No': False}).astype(bool)
DS1_data['Consumer consent provided?'] = DS1_data['Consumer consent provided?'].replace(
    {'Consent provided': True, '': False}).astype(bool)
DS1_data = DS1_data[pd.notnull(DS1_data['Consumer complaint narrative'])]  # Drops the missing rows with no  complaint.
DS1_data.drop(columns=['Sub-issue', "Product"])

db = sqlalchemy.create_engine('sqlite:///StaterData.db')
DS1_data.to_sql("mortgage complaints", db, if_exists="replace")
print("Database saved succesfully")
