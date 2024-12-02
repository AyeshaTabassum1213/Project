import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

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
    return features

def gibbs_sampling(data, num_topics, num_iterations):
    num_samples, num_features = data.shape
    
    # Initialize topic assignments randomly
    topics = np.random.randint(0, num_topics, size=(num_samples,))
    
    # Initialize counts for topic-word assignments
    topic_word_counts = np.zeros((num_topics, num_features))
    for i in range(num_samples):
        topic_word_counts[topics[i]] += data[i]
    
    # Initialize counts for topic assignments
    topic_counts = np.zeros(num_topics)
    for i in range(num_samples):
        topic_counts[topics[i]] += 1
    
    # Perform Gibbs sampling iterations
    for _ in range(num_iterations):
        for i in range(num_samples):
            # Decrement counts for current sample's topic assignment
            topic_word_counts[topics[i]] -= data[i]
            topic_counts[topics[i]] -= 1
            
            # Calculate the probability of assigning the current sample to each topic
            topic_probs = (topic_word_counts[:, i % num_features] + 1) / (np.sum(topic_word_counts, axis=1) + 2)
            
            # Sample a new topic for the current sample
            new_topic = np.random.choice(num_topics, p=topic_probs / topic_probs.sum())
            topics[i] = new_topic
                
            # Update counts for new topic assignment
            topic_word_counts[new_topic] += data[i]
            topic_counts[new_topic] += 1
    
    return topics

def evaluate_model(true_labels, predictions, class_label):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, pos_label=class_label)
    recall = recall_score(true_labels, predictions, pos_label=class_label)
    f1 = f1_score(true_labels, predictions, pos_label=class_label)
    return accuracy, precision, recall, f1

def detect_fake_news(file_path):
    df = load_dataset(file_path)
    df = preprocess_data(df)
    features = extract_features(df)
    topics = gibbs_sampling(features[:, 3:], num_topics=2, num_iterations=10)
    
    # Reshape topics array if it's 1D
    if topics.ndim == 1:
        topics = topics.reshape(-1, 1)
    
    predictions = np.argmax(topics, axis=1)
    df['Predictions'] = predictions
    fake_news_df = df[df['Predictions'] == 1]
    true_news_df = df[df['Predictions'] == 0]
    fake_news_df.to_csv("fake_news.csv", index=False)
    true_news_df.to_csv("true_news.csv", index=False)
    print("\nFake news saved to 'fake_news.csv'")
    print("True news saved to 'true_news.csv'")
    return None, None, df['fake_status'].values

def main():
    file_path = "facebook.csv"
    _, _, true_labels = detect_fake_news(file_path)
    predictions = np.random.randint(0, 2, len(true_labels))  # Dummy predictions for illustration
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
