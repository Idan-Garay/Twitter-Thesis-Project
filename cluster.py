from functionss import preprocess_dataset
from sklearn.model_selection import KFold
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

pd.options.mode.chained_assignment = None


def cluster(dataset, best_k = 1):
    # corpus = data['Clean']
    corpus = dataset['Clean'].astype('str')

    # tfidf = TfidfVectorizer(
    #     min_df = 5,
    #     max_df = 0.95,
    # )

    tfidf = CountVectorizer(
        min_df = 9,
        # max_df = 0.84,
        max_df = 0.90,
    )

    tfidf.fit(corpus)
    textMatrix = tfidf.transform(corpus)

    print("Preprocessed Dataset size: {}".format(corpus.size))

    X = textMatrix.toarray()
    
    if best_k != 1:
        final_model = AgglomerativeClustering(n_clusters=best_k)
        final_model.fit(X)
        return final_model, corpus, X, tfidf, best_k


    kf = KFold(n_splits=5)

    num_clusters = [3,4,5,6,7,8,9,10,11,12,13]

    # Placeholder for scores
    scores = []

    # Loop over the number of clusters
    for k in num_clusters:
        sil_scores = []
        
        # Loop over the folds
        for train_index, test_index in kf.split(X):
            # Split data into train and test sets
            X_train, X_test = X[train_index], X[test_index]
            
            # Fit the clustering model
            model = AgglomerativeClustering(n_clusters=k, linkage='ward')
            model.fit(X_train)
            
            # Predict the clusters for the test set
            y_pred = model.fit_predict(X_test)
            
            # Compute the silhouette score
            score = silhouette_score(X_test, y_pred, random_state = 42)
            
            sil_scores.append(score)
            
        # Store the average silhouette score
        avg_score = np.mean(sil_scores)
        scores.append(avg_score)

    # Find the number of clusters with the highest average silhouette score
    best_k = num_clusters[np.argmax(scores)]

    # Fit the final model with the best number of clusters
    final_model = AgglomerativeClustering(n_clusters=best_k)
    final_model.fit(X)

    print("Final Agglomerative Model's Clusters: {}".format(final_model.n_clusters_))
    print()
    table_x = pd.DataFrame({
        'number_of_clusters': num_clusters,
        'silhouette_score': scores
    })

    print(table_x)
    return final_model, corpus, X, tfidf, best_k

def get_tweets_clusters(cluster_model, dataset):
    # categorize dataset[clean] by cluster label
    tweets_clusters = [[] for x in range(cluster_model.n_clusters_)]
    
    for index, tweet in enumerate(dataset['Clean']):
        tweets_clusters[cluster_model.labels_[index]].append(tweet)
    return tweets_clusters
