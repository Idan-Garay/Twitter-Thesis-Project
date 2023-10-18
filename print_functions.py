import numpy as np

clusters_text = []

def print_process(model, df, corpus_embeddings, tfidf):
  cluster_labels = model.labels_

  # Create a dictionary to store the words in each cluster
  clusters = {i: [] for i in range(model.n_clusters)}

  print('')
  # Iterate over the data and the cluster labels
  for i, label in enumerate(cluster_labels):
      # Split the text into words
      words = df.iloc[i].split()
      # Add the words to the appropriate cluster
      clusters[label].extend(words)

  for i in range(model.n_clusters):
    cluster_indices = np.where(model.labels_ == i)[0]
    cluster_features = corpus_embeddings[cluster_indices]
    feature_means = np.mean(cluster_features, axis=0)
    top_features = np.argsort(feature_means)[-10:]
    clusters_text.append([tfidf.get_feature_names_out()[i] for i in top_features])
    print("Cluster", i, ":", clusters_text[i])

  print()



from wordcloud import WordCloud 
import matplotlib.pyplot as plt

# text = clusters_text[1]

# def word_cloud():
#     wordcloud = WordCloud(background_color='white', width=400, height=450, margin=4, prefer_horizontal=1.5).generate(" ".join(text))

#     # Display the generated image:
#     plt.figure(figsize=(8, 12))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis("off")
#     plt.show()