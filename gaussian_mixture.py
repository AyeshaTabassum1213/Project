import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.cluster import contingency_matrix
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df.fillna(0, inplace=True)
    label_encoder = LabelEncoder()
    df['fake_status'] = label_encoder.fit_transform(df['fake_status'])
    return df

def extract_features(df):
    numerical_features = ['likes', 'shares', 'comments']
    text_features = 'post_text'
    text_pipeline = make_pipeline(
        CountVectorizer(max_features=1000, stop_words='english')
    )
    feature_pipeline = ColumnTransformer(
        [('numerical', 'passthrough', numerical_features),
         ('text', text_pipeline, text_features)]
    )
    features = feature_pipeline.fit_transform(df)
    return features.toarray(), df['fake_status'].values  # Convert sparse to dense

def train_classifier(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("Supervised Classification Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    
    return model, X_test, y_test

def gmm_clustering(features, true_labels, num_components=2):
    gmm = GaussianMixture(n_components=num_components, random_state=42)
    gmm.fit(features)
    cluster_labels = gmm.predict(features)
    
    # Extract GMM parameters
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_
    
    print("\nGMM Parameters:")
    print("Means:\n", means)
    print("\nCovariances:\n", covariances)
    print("\nWeights:\n", weights)
    
    # Purity Score Calculation
    contingency = contingency_matrix(true_labels, cluster_labels)
    purity_score = np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)
    
    # Adjusted Rand Score
    adjusted_rand = adjusted_rand_score(true_labels, cluster_labels)
    
    print("\nGMM Clustering Metrics:")
    print("Purity Score:", purity_score)
    print("Adjusted Rand Score:", adjusted_rand)
    
    return cluster_labels

def evaluate_clustering(cluster_labels, true_labels):
    accuracy = accuracy_score(true_labels, cluster_labels)
    precision = precision_score(true_labels, cluster_labels)
    recall = recall_score(true_labels, cluster_labels)
    f1 = f1_score(true_labels, cluster_labels)
    
    print("\nClustering Evaluation Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    
def main():
    file_path = "facebook.csv"
    df = load_dataset(file_path)
    df = preprocess_data(df)
    features, labels = extract_features(df)
    
    model, X_test, y_test = train_classifier(features, labels)
    cluster_labels = gmm_clustering(X_test, y_test)
    evaluate_clustering(cluster_labels, y_test)
    
    # Plot the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(cluster_labels)), cluster_labels, c=cluster_labels, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.title('GMM Cluster Distribution')
    plt.xlabel('Data Point')
    plt.ylabel('Cluster')
    plt.colorbar(label='Cluster')
    plt.show()

if __name__ == "__main__":
    main()
