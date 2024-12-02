import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df.fillna(0, inplace=True)
    label_encoder = LabelEncoder()
    df['fake_status'] = label_encoder.fit_transform(df['fake_status'])
    return df

def extract_features(df):
    # Numerical features
    numerical_features = ['likes', 'shares', 'comments']
    
    # Text features
    text_features = 'post_text'
    
    # Create a pipeline for text feature extraction
    text_pipeline = make_pipeline(
        TfidfVectorizer(max_features=1000, stop_words='english')
    )
    
    # Combine numerical and text features
    feature_pipeline = ColumnTransformer(
        [('numerical', 'passthrough', numerical_features),
         ('text', text_pipeline, text_features)]
    )
    
    # Extract features
    features = feature_pipeline.fit_transform(df)
    
    return features

def train_model(features):
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(features)
    return kmeans

def evaluate_model(model, features, true_labels, class_label):
    predictions = model.predict(features)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, pos_label=class_label)
    recall = recall_score(true_labels, predictions, pos_label=class_label)
    f1 = f1_score(true_labels, predictions, pos_label=class_label)
    
    return accuracy, precision, recall, f1, predictions

def detect_fake_news(file_path):
    df = load_dataset(file_path)
    df = preprocess_data(df)
    features = extract_features(df)
    model = train_model(features)
    
    # Predictions
    predictions = model.predict(features)
    
    # Save fake news and true news in separate files
    df['Predictions'] = predictions
    fake_news_df = df[df['Predictions'] == 1]
    true_news_df = df[df['Predictions'] == 0]
    
    fake_news_df.to_csv("fake_news.csv", index=False)
    true_news_df.to_csv("true_news.csv", index=False)
    print("\nFake news saved to 'fake_news.csv'")
    print("True news saved to 'true_news.csv'")
    
    return model, features, df['fake_status'].values

def main():
    file_path = "facebook.csv"
    model, features, true_labels = detect_fake_news(file_path)

    accuracy_fake, precision_fake, recall_fake, f1_score_fake, _ = evaluate_model(model, features, true_labels, class_label=1)
    accuracy_true, precision_true, recall_true, f1_score_true, _ = evaluate_model(model, features, true_labels, class_label=0)

    print("Metrics for Fake News:")
    print("Accuracy:", accuracy_fake)
    print("Precision:", precision_fake)
    print("Recall:", recall_fake)
    print("F1-score:", f1_score_fake)

    print("\nMetrics for True News:")
    print("Accuracy:", accuracy_true)
    print("Precision:", precision_true)
    print("Recall:", recall_true)
    print("F1-score:", f1_score_true)

if __name__ == "__main__":
    main()