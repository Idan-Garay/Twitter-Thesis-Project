# for macos: python3 -m pip install tweet-preprocessor snscrape contractions lingua-language-detector langdetect advertools wordcloud vaderSentiment  numpy advertools contractions nltk scikit-learn
# for windows: python -m pip install tweet-preprocessor snscrape contractions lingua-language-detector langdetect advertools wordcloud vaderSentiment  numpy advertools contractions nltk scikit-learn
from classification import sentiment_classification, annotate_corpus_for_sentiments
import pandas as pd
from cluster import cluster, get_tweets_clusters
from print_functions import print_process
from sklearn.cluster import AgglomerativeClustering
from functionss import preprocess_dataset 

# ---------------------------------- NOTICE ---------------------------------- #
# the library snscrape's twitter scraper is not working anymore due to twitter's new policy on scraping
# refer to https://github.com/JustAnotherArchivist/snscrape/issues/996#issuecomment-1615195000
# refer to https://twitter.com/elonmusk/status/1675187969420828672 

# from data_gather import fetch_dataset_from_twitter
# tweets = fetch_dataset_from_twitter() # 404 error due to twitter's new policy on scraping
# ---------------------------------- END NOTICE ---------------------------------- #

dataset = pd.read_csv('dataset.csv') 
dataset = preprocess_dataset(dataset)

# assumes dataset['Clean'] as corpus for cluster method
# cluster method finds the best_k while assuming a default best_k = 1
cluster_model, corpus, X, tfidf, best_k = cluster(dataset, best_k = 1) 

# best_k = 4
final_model = AgglomerativeClustering(n_clusters=best_k)
final_model.fit(X)  
print_process(final_model, corpus, X, tfidf)

# annotate each tweet with a sentiment using SentimentIntensityAnalyzer
dataset = annotate_corpus_for_sentiments(dataset)

# segragate tweets into their respective cluster
tweets_clusters = get_tweets_clusters(final_model, dataset)

# classify each cluster's sentiment
clusters_sentiment = sentiment_classification(tweets_clusters)
print(clusters_sentiment)

