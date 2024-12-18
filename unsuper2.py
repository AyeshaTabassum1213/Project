# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from wordcloud import WordCloud

# Read the dataset
news_data = pd.read_table('./news_data.tsv', delimiter='\t')

news_data.head()

# Drop un-necessary columns
news_data.drop(columns=['2635.json', 'abortion', 'dwayne-bohac', 'State representative', 'Texas', 'republican', '0', '1', '0.1', '0.2', '0.3', 'a mailer'], inplace=True)

# Rename columns
news_data.columns = ['label', 'news_text']

news_data.head()

# Check for null values
print("Null values in dataset:")
print(news_data.isnull().sum())

each_label_count = news_data['label'].value_counts()
each_label_count

news_data_half_true_only = news_data[news_data['label'] == 'half-true']
news_data_false_only = news_data[news_data['label'] == 'false']
news_data_mostly_true_only = news_data[news_data['label'] == 'mostly-true']
news_data_true_only = news_data[news_data['label'] == 'true']
news_data_barely_true_only = news_data[news_data['label'] == 'barely-true']
news_data_pants_fire_only = news_data[news_data['label'] == 'pants-fire']

# Sample 830 data points for each label
news_data_half_true_only_sample = news_data_half_true_only.sample(830, random_state=42)
news_data_false_only_sample = news_data_false_only.sample(830, random_state=42)
news_data_mostly_true_only_sample = news_data_mostly_true_only.sample(830, random_state=42)
news_data_true_only_sample = news_data_true_only.sample(830, random_state=42)
news_data_barely_true_only_sample = news_data_barely_true_only.sample(830, random_state=42)
news_data_pants_fire_only_sample = news_data_pants_fire_only.sample(830, random_state=42)


# Concatenate the sampled data
news_data = pd.concat([news_data_half_true_only_sample, news_data_false_only_sample, news_data_mostly_true_only_sample, news_data_true_only_sample, news_data_barely_true_only_sample, news_data_pants_fire_only_sample], axis=0)
# Shuffle the rows of the DataFrame
news_data = news_data.sample(frac=1.0, random_state=42)


news_data.head()

# Preprocessing the news texts
nltk.download('stopwords')
nltk.download('punkt')
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    word_arr = []
    for i in text:
        if i.isalnum():
            word_arr.append(i)
    text = word_arr[:]
    word_arr.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            word_arr.append(i)
    text = word_arr[:]
    word_arr.clear()
    for i in text:
        word_arr.append(ps.stem(i))
    return " ".join(word_arr)

news_data['preprocessed_news_text'] = news_data['news_text'].apply(preprocess_text)
news_data.head()


# Concatenate preprocessed text and generate word cloud
preprocessed_text = ' '.join(news_data['preprocessed_news_text'])
wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(preprocessed_text)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Drop unprocessed news_text and label columns
news_data.drop(columns=['news_text', 'label'], inplace=True)


# Initializing TfidfVectorizer
vectorizer = TfidfVectorizer()

# Vectorizing the pre-processed news texts
X = vectorizer.fit_transform(news_data['preprocessed_news_text']).toarray()


# Applying Principal Component Analysis (PCA) to reduce dimensionality
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)


# Visualize PCA
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap='viridis')
plt.title('Data Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Color Intensity')
plt.show()


# Finding optimal number of clusters for K-Means using elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)
optimal_kmeans_clusters = np.argmin(wcss) + 1

optimal_kmeans_clusters

# Plotting the elbow point graph
sns.set()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Training K-Means with optimal number of clusters
optimal_kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans_labels = optimal_kmeans.fit_predict(X_pca)

# Finding optimal number of clusters for K-Medoids using elbow method
wcss_kmedoids = []
for i in range(1, 11):
    kmedoids = KMedoids(n_clusters=i, init='k-medoids++', random_state=42)
    kmedoids.fit(X_pca)
    wcss_kmedoids.append(kmedoids.inertia_)
optimal_kmedoids_clusters = np.argmin(wcss_kmedoids) + 1


# Plotting the elbow point graph for K-Medoids
plt.plot(range(1, 11), wcss_kmedoids)
plt.title('The Elbow Point Graph for K-Medoids')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Training K-Medoids with optimal number of clusters
optimal_kmedoids = KMedoids(n_clusters=5, init='k-medoids++', random_state=42)
kmedoids_labels = optimal_kmedoids.fit_predict(X_pca)

# Finding optimal number of clusters for Agglomerative Clustering using dendrogram
Z = shc.linkage(X_pca, method='ward')
plt.figure(figsize=(10, 7))
plt.title('News Dendogram')
dend = shc.dendrogram(shc.linkage(X_pca, method='ward'))
plt.show()


# Training Agglomerative Clustering with optimal number of clusters
optimal_agglomerative = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
hierarchical_labels = optimal_agglomerative.fit_predict(X_pca)

# Finding optimal number of components for Gaussian Mixture Model (GMM) using BIC
lowest_bic = np.infty
bic = []
n_components_range = range(1, 6)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=42)
        gmm.fit(X_pca)
        bic.append(gmm.bic(X_pca))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            optimal_gmm = gmm


import itertools
# Plotting the BIC values for GMM
plt.figure(figsize=(8, 6))
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])
bars = []
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):(i + 1) * len(n_components_range)], width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([min(bic) * 1.01 - .01 * max(bic), max(bic)])
plt.title('BIC score per model')
xpos = np.mod(np.argmin(bic), len(n_components_range)) + .65 + .2 * np.floor(np.argmin(bic) / len(n_components_range))
plt.text(xpos, min(bic) * 0.97 + .03 * max(bic), '*', fontsize=14)
plt.xlabel('Number of components')
plt.legend([b[0] for b in bars], cv_types)
plt.show()

# Training GMM with optimal number of components
gmm_labels = optimal_gmm.predict(X_pca)


# Finding optimal parameters for DBSCAN using silhouette score
best_silhouette_score = -1
optimal_eps = 0
optimal_min_samples = 0
best_dbscan_labels = None
silhouette_scores_dbscan = []  # Store silhouette scores for each combination
for eps in np.arange(0.1, 1.0, 0.1):
    for min_samples in range(2, 11):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(X_pca)

        # Skip calculation if only one label is present
        if len(np.unique(dbscan_labels)) == 1:
            continue

        silhouette_score_dbscan = silhouette_score(X_pca, dbscan_labels)
        silhouette_scores_dbscan.append(silhouette_score_dbscan)  # Store silhouette score

        if silhouette_score_dbscan > best_silhouette_score:
            best_silhouette_score = silhouette_score_dbscan
            optimal_eps = eps
            optimal_min_samples = min_samples
            best_dbscan_labels = dbscan_labels



# Plotting the silhouette score for DBSCAN
plt.plot(range(1, len(silhouette_scores_dbscan) + 1), silhouette_scores_dbscan)
plt.title('Silhouette Score for DBSCAN')
plt.xlabel('Number of Combinations')
plt.ylabel('Silhouette Score')
plt.show()

# Training DBSCAN with optimal parameters
optimal_dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
dbscan_labels = optimal_dbscan.fit_predict(X_pca)

# Visualize the results
plt.figure(figsize=(10, 7))

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')
plt.show()


plt.figure(figsize=(10,7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmedoids_labels, cmap='viridis')
plt.title('K-Medoids Clustering')
plt.show()


plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='viridis')
plt.title('Agglomerative Clustering')
plt.show()


plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='viridis')
plt.title('GMM Clustering')
plt.show()


plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.show()



# Calculate silhouette score for KMeans
silhouette_score_kmeans = silhouette_score(X_pca, kmeans_labels)

# Calculate silhouette score for KMedoids
silhouette_score_kmedoid = silhouette_score(X_pca, kmedoids_labels)

# Calculate silhouette score for Agglomerative Hierarchical Clustering
silhouette_score_hierarchy = silhouette_score(X_pca, hierarchical_labels)

# Calculate silhouette score for GMM
silhouette_score_gmm = silhouette_score(X_pca, gmm_labels)

# Calculate silhouette score for DBSCAN
silhouette_score_dbscan = silhouette_score(X_pca, best_dbscan_labels)


# Create a DataFrame
metrics_df = pd.DataFrame({
    'Algorithm': ['KMeans', 'KMedoids', 'Agglomerative Hierarchical', 'GMM', 'DBSCAN'],
    'Silhouette Score': [silhouette_score_kmeans, silhouette_score_kmedoid, silhouette_score_hierarchy, silhouette_score_gmm, silhouette_score_dbscan]
})

# Display the DataFrame
print(metrics_df)

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(metrics_df['Algorithm'], metrics_df['Silhouette Score'], color='skyblue')
plt.title('Silhouette Score for Clustering Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('Silhouette Score')
plt.xticks(rotation=45)
plt.show()

