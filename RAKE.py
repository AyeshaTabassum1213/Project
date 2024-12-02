import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rake_nltk import Rake
import matplotlib.pyplot as plt

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df.fillna('', inplace=True)  # Fill missing values with empty strings for text data
    label_encoder = LabelEncoder()
    df['fake_status'] = label_encoder.fit_transform(df['fake_status'])
    return df

def extract_keywords(df):
    rake = Rake()
    def rake_keywords(text):
        rake.extract_keywords_from_text(text)
        keywords = rake.get_ranked_phrases()
        return ' '.join(keywords) if keywords else 'no_keywords'

    df['keywords'] = df['post_text'].apply(rake_keywords)
    return df

def extract_features(df):
    numerical_features = ['likes', 'shares', 'comments']
    text_features = 'keywords'
    
    # Standardize numerical features
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    text_pipeline = make_pipeline(
        CountVectorizer(max_features=1000, stop_words='english')
    )
    feature_pipeline = ColumnTransformer(
        [('numerical', 'passthrough', numerical_features),
         ('text', text_pipeline, text_features)]
    )
    features = feature_pipeline.fit_transform(df)
    return features

def apply_classifier(features, labels):
    clf = LogisticRegression(max_iter=200)  # Increase max_iter to 200
    clf.fit(features, labels)
    predictions = clf.predict(features)
    return predictions

def evaluate_model(true_labels, predictions, class_label):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, pos_label=class_label)
    recall = recall_score(true_labels, predictions, pos_label=class_label)
    f1 = f1_score(true_labels, predictions, pos_label=class_label)
    return accuracy, precision, recall, f1

def detect_fake_news(file_path):
    df = load_dataset(file_path)
    df = preprocess_data(df)
    df = extract_keywords(df)
    features = extract_features(df)
    labels = df['fake_status'].values
    predictions = apply_classifier(features, labels)
    
    df['Predictions'] = predictions
    fake_news_df = df[df['Predictions'] == 1]
    true_news_df = df[df['Predictions'] == 0]
    fake_news_df.to_csv("fake_news.csv", index=False)
    true_news_df.to_csv("true_news.csv", index=False)
    print("\nFake news saved to 'fake_news.csv'")
    print("True news saved to 'true_news.csv'")
    
    # Plot the distribution of predictions
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(predictions)), predictions, c=predictions, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.title('Distribution of Predictions')
    plt.xlabel('Data Point')
    plt.ylabel('Prediction')
    plt.colorbar(label='Prediction')
    plt.show()
    
    return predictions, df['fake_status'].values

def main():
    file_path = "facebook.csv"
    predictions, true_labels = detect_fake_news(file_path)
    accuracy_fake, precision_fake, recall_fake, f1_score_fake = evaluate_model(true_labels, predictions, class_label=1)
    accuracy_true, precision_true, recall_true, f1_score_true = evaluate_model(true_labels, predictions, class_label=0)
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
