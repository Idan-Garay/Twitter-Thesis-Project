# python3 -m pip install tweet-preprocessor snscrape contractions lingua-language-detector langdetect advertools wordcloud vaderSentiment
# python3 -m pip install langdetect numpy advertools contractions nltk scikit-learn snscrape               
from classification import sentiment_classification, annotate_tweet
from functionss import  preprocess_dataset 
import pandas as pd
from cluster import cluster, get_tweets_clusters
from print_functions import print_process
from sklearn.cluster import AgglomerativeClustering
from data_gather import fetch_dataset_from_twitter


dataset = pd.read_csv('preprocessed.csv')


cluster_model, corpus, X, tfidf, best_k = cluster(dataset, best_k =1)

# best_k = 4
final_model = AgglomerativeClustering(n_clusters=best_k)
final_model.fit(X)  
print_process(final_model, corpus, X, tfidf)


dataset = dataset.dropna(subset=['Clean'])
dataset.reset_index(drop=True)
dataset['sentiment'] = dataset['Clean'].apply(lambda text: annotate_tweet(text))

tweets_clusters = get_tweets_clusters(final_model, dataset)
clusters_sentiment = sentiment_classification(tweets_clusters)
print(clusters_sentiment)

