import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df.fillna('', inplace=True)
    return df

def extract_features(df):
    text_features = 'post_text'
    numerical_features = ['likes', 'shares', 'comments']

    # Text feature extraction
    text_vectorizer = CountVectorizer(stop_words='english')
    text_features_matrix = text_vectorizer.fit_transform(df[text_features])

    # Numerical feature extraction and scaling
    numerical_data = df[numerical_features].values
    scaler = MinMaxScaler()  # Use MinMaxScaler to avoid negative values
    numerical_features_scaled = scaler.fit_transform(numerical_data)

    # Combine text and numerical features
    combined_features = np.hstack([text_features_matrix.toarray(), numerical_features_scaled])
    return combined_features, text_vectorizer

def apply_lda(features):
    lda = LatentDirichletAllocation(n_components=2, random_state=42)
    lda_topics = lda.fit_transform(features)
    return lda, lda_topics

def assign_topics(df, lda_topics):
    topic_labels = np.argmax(lda_topics, axis=1)
    return topic_labels

def map_topics_to_labels(df, topic_labels):
    true_labels = df['fake_status']
    topic_to_label = {}
    for topic in set(topic_labels):
        labels = [true_labels[i] for i in range(len(true_labels)) if topic_labels[i] == topic]
        if labels:
            topic_to_label[topic] = pd.Series(labels).mode().iloc[0]
    
    predicted_labels = [topic_to_label.get(topic, -1) for topic in topic_labels]
    return predicted_labels

def evaluate_topics(df, predicted_labels):
    true_labels = df['fake_status']
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, pos_label=1, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, pos_label=1, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, pos_label=1, zero_division=0)
    
    return accuracy, precision, recall, f1

def visualize_topics(lda, vectorizer):
    words = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_word_indices = topic.argsort()[:-10 - 1:-1]
        top_words = [words[i] for i in top_word_indices]
        top_weights = topic[top_word_indices]

        print(f"Topic #{topic_idx}:")
        print(" ".join(top_words))

        plt.figure(figsize=(10, 6))
        plt.barh(top_words, top_weights, color='skyblue')
        plt.xlabel('Weight')
        plt.title(f'Topic #{topic_idx}')
        plt.gca().invert_yaxis()  # Invert y-axis to have the highest weights at the top
        plt.show()


def main():
    file_path = "facebook.csv"
    df = load_dataset(file_path)
    df = preprocess_data(df)
    
    features, text_vectorizer = extract_features(df)
    lda, lda_topics = apply_lda(features)
    
    visualize_topics(lda, text_vectorizer)
    
    topic_labels = assign_topics(df, lda_topics)
    predicted_labels = map_topics_to_labels(df, topic_labels)
    accuracy, precision, recall, f1 = evaluate_topics(df, predicted_labels)
    
    print("LDA Topic Modeling Performance Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    
    # Save the topic assignment results
    df['Topic'] = topic_labels
    df['Predicted_Label'] = predicted_labels
    df.to_csv("lda_topic_results.csv", index=False)
    print("LDA topic results saved to 'lda_topic_results.csv'")

if __name__ == "__main__":
    main()
