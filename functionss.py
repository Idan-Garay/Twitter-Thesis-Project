from langdetect import detect
import numpy as np
import advertools as adv
import contractions
import preprocessor as p
import nltk
from nltk.stem import WordNetLemmatizer

# pd.set_option('display.max_colwidth', None)
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

def replaceProcess(text):
  specialSymbols = ['.', '?', ',', '%', '!', ';', '|', '$', '&amp', '-', '#', '*', '_', ':', '(', ')', '*', '"', '/']
  for symbol in specialSymbols:
    text = text.replace(symbol, " ")
  return text

def preprocess_tweet(text):
  text = p.clean(text) # goes through all settings unless specified
  text = replaceProcess(text)
  return text

def preprocess_dataset(dataset):
  tagalog_stopwords = adv.stopwords['tagalog']
  tagalog_stopwords.add('yung')
  tagalog_stopwords.add('jan')
  tagalog_stopwords.add('najud')
  tagalog_stopwords.add('hoy')
  tagalog_stopwords.add('tapos')
  tagalog_stopwords.add('man')
  tagalog_stopwords.add('mag')
  tagalog_stopwords.add('grabe')
  tagalog_stopwords.add('mahal')
  english_stopwords = adv.stopwords['english']
  tagalog_stopwords.add('hahaha')
  tagalog_stopwords.add('percent')
  tagalog_stopwords.add('today')
  tagalog_stopwords.add('tomorrow')
  tagalog_stopwords.add('yesterday')
  contractions.add('youre', 'you are')
  

  if (dataset['Clean'].empty == False):
    return dataset

  # Remove Rows with NaN value in Clean column
  dataset.dropna(subset=['Text'], inplace=True)
  dataset['Text'] = dataset['Text'].str.lower()
  # tweet preprocess
  dataset['Clean'] = dataset['Text'].apply(lambda text: preprocess_tweet(text))
  # Replace empty cells to NaN
  dataset['Clean'] = dataset['Clean'].replace(' ', np.nan,)
  # Remove Rows with NaN value in Clean column
  dataset.dropna(subset=['Clean'], inplace=True)
  dataset.reset_index(drop=True)
  # Remove Duplicates
  dataset.drop_duplicates(subset=['Clean'])
  # filter tweet from words with less than 3 characters
  dataset['Clean_Text_Length'] = dataset['Clean'].apply(lambda text: len(text))
  dataset = dataset[dataset['Clean_Text_Length'] > 3]
  # Assign tweet with detected language
  dataset['Language'] = dataset['Clean'].apply(lambda text: detect(text))
  # Filter non-English tweets
  dataset = dataset[dataset['Language'] == 'en']
  dataset.reset_index(drop=True)

  # remove tagalog stopwords
  dataset['Clean'] = dataset['Clean'].apply(lambda text: ' '.join([word for word in text.split() if word not in tagalog_stopwords]))
  dataset.reset_index(drop=True)
  
  # expand contractions
  dataset['Clean'] = dataset['Clean'].apply(lambda text: contractions.fix(text))
  # remove english words
  dataset['Clean'] = dataset['Clean'].apply(lambda text: ' '.join([word for word in text.split() if word not in english_stopwords]))

  
  #lemmatize
  dataset['Clean'] = dataset['Clean'].apply(lambda text: ' '.join([lemmatizer.lemmatize(word) for word in text.split()]))

  # filter tweet from words with less than 3 characters
  dataset['Clean'] = dataset['Clean'].apply(lambda text: ' '.join([word for word in text.split() if len(word) > 4]))
  # remove tagalog stopwords
  dataset['Clean'] = dataset['Clean'].apply(lambda text: ' '.join([word for word in text.split() if word not in tagalog_stopwords]))
  
  # remove empty and duplicates
  dataset = dataset.drop_duplicates(subset=['Clean'])
  dataset.reset_index(drop=True)
  dataset = dataset.dropna(subset=['Clean'])
  dataset.reset_index(drop=True)
  dataset = dataset[dataset['Clean'] != 'nan']

  return dataset