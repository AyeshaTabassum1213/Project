import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import umap
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df.fillna(0, inplace=True)
    return df

def extract_features(df):
    numerical_features = ['likes', 'shares', 'comments']
    text_features = 'post_text'
    text_pipeline = make_pipeline(
        CountVectorizer(max_features=500, stop_words='english')  # Reduced max_features for faster processing
    )
    feature_pipeline = ColumnTransformer(
        [('numerical', 'passthrough', numerical_features),
         ('text', text_pipeline, text_features)]
    )
    features = feature_pipeline.fit_transform(df)
    return features

def apply_umap(features):
    umap_model = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1, random_state=42)  # Adjusted parameters for faster computation
    embedding = umap_model.fit_transform(features)
    return embedding

def evaluate_clustering(df, predictions):
    true_labels = df['fake_status']
    predicted_labels = predictions
    
    # Assign clusters to true labels based on majority class
    cluster_0_label = true_labels[predicted_labels == 0].mode()[0]
    cluster_1_label = true_labels[predicted_labels == 1].mode()[0]
    
    cluster_labels = np.where(predicted_labels == 0, cluster_0_label, cluster_1_label)
    
    accuracy = accuracy_score(true_labels, cluster_labels)
    precision = precision_score(true_labels, cluster_labels, zero_division=0)
    recall = recall_score(true_labels, cluster_labels, zero_division=0)
    f1 = f1_score(true_labels, cluster_labels, zero_division=0)
    
    return accuracy, precision, recall, f1

def detect_fake_news(file_path):
    df = load_dataset(file_path)
    df = preprocess_data(df)
    features = extract_features(df)
    embedding = apply_umap(features)
    
    # Apply Agglomerative Clustering
    agglomerative_clustering = AgglomerativeClustering(n_clusters=2)
    predictions = agglomerative_clustering.fit_predict(embedding)
    
    df['Cluster'] = predictions
    cluster_0 = df[df['Cluster'] == 0]
    cluster_1 = df[df['Cluster'] == 1]
    cluster_0.to_csv("cluster_0.csv", index=False)
    cluster_1.to_csv("cluster_1.csv", index=False)
    print("\nCluster 0 saved to 'cluster_0.csv'")
    print("Cluster 1 saved to 'cluster_1.csv'")
    
    # Plot the distribution of predicted clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=df['Cluster'].values, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.title('UMAP Projection of Text Data')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.colorbar(label='Cluster')
    plt.show()
    
    return df, predictions

def main():
    file_path = "facebook.csv"
    df, predictions = detect_fake_news(file_path)
    accuracy, precision, recall, f1 = evaluate_clustering(df, predictions)
    print("Clustering Performance Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

if __name__ == "__main__":
    main()
