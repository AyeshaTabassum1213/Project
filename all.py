import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import umap
import numpy as np
import matplotlib.pyplot as plt
from rake_nltk import Rake
from gensim.summarization import keywords
import networkx as nx
from networkx.algorithms import community

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

def detect_anomalies_with_isolation_forest(features):
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    predictions = iso_forest.fit_predict(features)
    return predictions

def detect_anomalies_with_one_class_svm(features):
    svm = OneClassSVM(gamma='scale', nu=0.1)
    predictions = svm.fit_predict(features)
    return predictions

def evaluate_anomalies(df, predictions):
    true_labels = df['fake_status']
    cluster_labels = np.where(predictions == 1, 0, 1)  # Anomalies as fake news

    accuracy = accuracy_score(true_labels, cluster_labels)
    precision = precision_score(true_labels, cluster_labels, zero_division=0)
    recall = recall_score(true_labels, cluster_labels, zero_division=0)
    f1 = f1_score(true_labels, cluster_labels, zero_division=0)

    return accuracy, precision, recall, f1

def extract_keywords_rake(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()

def extract_keywords_textrank(text):
    return keywords(text).split('\n')

def detect_fake_news(file_path):
    df = load_dataset(file_path)
    df = preprocess_data(df)
    features = extract_features(df)
    embedding = apply_umap(features)
    
    # Anomaly detection
    isolation_forest_predictions = detect_anomalies_with_isolation_forest(embedding)
    one_class_svm_predictions = detect_anomalies_with_one_class_svm(embedding)
    
    # Keyword extraction
    df['RAKE Keywords'] = df['post_text'].apply(extract_keywords_rake)
    df['TextRank Keywords'] = df['post_text'].apply(extract_keywords_textrank)
    
    # Community detection
    G = nx.Graph()
    for idx, row in df.iterrows():
        keywords_list = row['RAKE Keywords'] + row['TextRank Keywords']
        for keyword in keywords_list:
            G.add_edge(idx, keyword)
    
    communities = community.greedy_modularity_communities(G)
    
    df['Community'] = -1
    for i, com in enumerate(communities):
        for node in com:
            if isinstance(node, int):
                df.at[node, 'Community'] = i
    
    return df, isolation_forest_predictions, one_class_svm_predictions

def main():
    file_path = "facebook.csv"
    df, iso_forest_predictions, svm_predictions = detect_fake_news(file_path)
    
    # Evaluate Isolation Forest
    iso_accuracy, iso_precision, iso_recall, iso_f1 = evaluate_anomalies(df, iso_forest_predictions)
    print("Isolation Forest Performance Metrics:")
    print("Accuracy:", iso_accuracy)
    print("Precision:", iso_precision)
    print("Recall:", iso_recall)
    print("F1-score:", iso_f1)

    # Evaluate One-Class SVM
    svm_accuracy, svm_precision, svm_recall, svm_f1 = evaluate_anomalies(df, svm_predictions)
    print("One-Class SVM Performance Metrics:")
    print("Accuracy:", svm_accuracy)
    print("Precision:", svm_precision)
    print("Recall:", svm_recall)
    print("F1-score:", svm_f1)
    
    # Save results
    df.to_csv("processed_facebook.csv", index=False)
    print("\nProcessed data saved to 'processed_facebook.csv'")

if __name__ == "__main__":
    main()
