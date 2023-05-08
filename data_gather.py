#@title Fetch Dataset
import numpy as np
import pandas as pd
import snscrape.modules.twitter as sntwitter

def fetch_dataset_from_files(isPreprocessed=False):
  dataset = []
  if isPreprocessed:
    dataset = dataset.read_csv('preprocess_dataset.csv') 
    return dataset

  pricehikedata = pd.read_csv('pricehike_uncleaned_dataset.csv')
  recessiondata = pd.read_csv('recession_uncleaned_dataset.csv')
  dataset = pd.read_csv('uncleaned_dataset.csv')

  dataset = dataset.append(recessiondata, ignore_index=True)
  dataset = dataset.append(pricehikedata, ignore_index=True)
  return dataset

def fetch_dataset_from_twitter():
  keywords = ['inflation', 'recession', 'price hike']

  tweets_list = []
  for keyword in keywords:
    search_query = keyword + ' until:2023-03-02 since:2022-02-01 geocode:12.71086,123.25827,667km'
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(search_query).get_items()):
        tweets_list.append([tweet.date, tweet.id, tweet.rawContent, tweet.user.username])
      
  # Creating a dataframe from the tweets list above
  tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
  return tweets_df

