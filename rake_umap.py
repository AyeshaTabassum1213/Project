import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import umap
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rake_nltk import Rake

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df.fillna(0, inplace=True)
    return df

def extract_keywords_rake(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return ' '.join(rake.get_ranked_phrases())

def extract_features(df):
    df['RAKE Keywords'] = df['post_text'].apply(extract_keywords_rake)
    text_features = 'RAKE Keywords'
    text_pipeline = make_pipeline(
        CountVectorizer(max_features=500, stop_words='english')
    )
    features = text_pipeline.fit_transform(df[text_features])
    return features

def apply_umap(features):
    umap_model = umap.UMAP(n_components=2, random_state=42)
    embedding = umap_model.fit_transform(features)
    return embedding

def cluster_and_evaluate(df, embedding):
    kmeans = KMeans(n_clusters=2, random_state=42)
    predictions = kmeans.fit_predict(embedding)

    df['Cluster'] = predictions

    # Assign clusters to true labels based on majority class
    cluster_0_label = df[df['Cluster'] == 0]['fake_status'].mode()[0]
    cluster_1_label = df[df['Cluster'] == 1]['fake_status'].mode()[0]

    predicted_labels = np.where(df['Cluster'] == 0, cluster_0_label, cluster_1_label)

    accuracy = accuracy_score(df['fake_status'], predicted_labels)
    precision = precision_score(df['fake_status'], predicted_labels, pos_label=1)
    recall = recall_score(df['fake_status'], predicted_labels, pos_label=1)
    f1 = f1_score(df['fake_status'], predicted_labels, pos_label=1)
    
    return accuracy, precision, recall, f1

def main():
    file_path = "facebook.csv"
    df = load_dataset(file_path)
    df = preprocess_data(df)
    features = extract_features(df)
    embedding = apply_umap(features)
    accuracy, precision, recall, f1 = cluster_and_evaluate(df, embedding)
    
    print("RAKE + UMAP + KMeans Clustering Performance Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

if __name__ == "__main__":
    main()
