import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

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

def detect_anomalies_isolation_forest(features):
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    predictions = iso_forest.fit_predict(features)
    return predictions

def evaluate_anomalies(true_labels, predictions):
    predictions_binary = np.where(predictions == -1, 1, 0)  # Convert to binary labels
    accuracy = accuracy_score(true_labels, predictions_binary)
    precision = precision_score(true_labels, predictions_binary, average='binary', pos_label=1)
    recall = recall_score(true_labels, predictions_binary, average='binary', pos_label=1)
    f1 = f1_score(true_labels, predictions_binary, average='binary', pos_label=1)
    return accuracy, precision, recall, f1

def main():
    file_path = "facebook.csv"
    df = load_dataset(file_path)
    df = preprocess_data(df)
    features = extract_features(df)
    predictions = detect_anomalies_isolation_forest(features)
    
    true_labels = df['fake_status'].values
    accuracy, precision, recall, f1 = evaluate_anomalies(true_labels, predictions)
    
    print("Isolation Forest Performance Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

if __name__ == "__main__":
    main()
