from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

def annotate_corpus_for_sentiments(dataset):
    dataset = dataset.dropna(subset=['Clean']) # drop rows with empty 'Clean' column
    dataset.reset_index(drop=True)
    dataset['sentiment'] = dataset['Clean'].apply(lambda text: annotate_tweet(text)) # errors due to empty 'Clean' column
    return dataset


def annotate_tweet(tweet):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_str = 'neutral'
    
    score = analyzer.polarity_scores(tweet)['compound']
    if score >= 0.05:
        sentiment_str = 'positive'
    elif score <= -0.05:
        sentiment_str = 'negative'
        
    return sentiment_str

def sentiment_classification(tweets_clusters):
    x = 0 
    analyzer = SentimentIntensityAnalyzer()
    cluster_compound_scores = []
    for tweet_cluster in tweets_clusters:
        total = 0
        for tweet in tweet_cluster:
            if tweet == 'nan':
                continue
            x += 1
            res = analyzer.polarity_scores(tweet)
            total += res['compound']
        cluster_compound_scores.append(total/len(tweet_cluster))
        
    ret = pd.DataFrame({
        'nth_cluster': range(0,len(tweets_clusters)),
        'sentiment_compound_score': cluster_compound_scores,
        'sentiment': [annotate(score) for score in cluster_compound_scores]
    })
    # elo['sentiment'] = ['positive', 'positive', 'negative', 'neutral']
    return ret

def annotate(score):
    sentiment_str = 'neutral'
    
    if score >= 0.05:
        sentiment_str = 'positive'
    elif score <= -0.05:
        sentiment_str = 'negative'
        
    return sentiment_str