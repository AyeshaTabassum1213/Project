import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
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
        CountVectorizer(max_features=500, stop_words='english')
    )
    feature_pipeline = ColumnTransformer(
        [('numerical', 'passthrough', numerical_features),
         ('text', text_pipeline, text_features)]
    )
    features = feature_pipeline.fit_transform(df)
    return features

def apply_umap(features):
    umap_model = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1, random_state=42)
    embedding = umap_model.fit_transform(features)
    return embedding

def detect_clusters_kmeans(embedding):
    kmeans = KMeans(n_clusters=2, random_state=42)
    predictions = kmeans.fit_predict(embedding)
    return predictions

def evaluate_clustering(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='binary', pos_label=1)
    recall = recall_score(true_labels, predicted_labels, average='binary', pos_label=1)
    f1 = f1_score(true_labels, predicted_labels, average='binary', pos_label=1)
    return accuracy, precision, recall, f1

def main():
    file_path = "facebook.csv"
    df = load_dataset(file_path)
    df = preprocess_data(df)
    features = extract_features(df)
    embedding = apply_umap(features)
    predictions = detect_clusters_kmeans(embedding)
    
    # Assuming 'fake_status' is the column with true labels
    true_labels = df['fake_status'].values
    
    accuracy, precision, recall, f1 = evaluate_clustering(true_labels, predictions)
    print("Clustering Performance Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

if __name__ == "__main__":
    main()
