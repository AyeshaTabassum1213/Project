import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import networkx as nx
import community as community_louvain
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    print("Dataset columns:", df.columns)  # Debugging line
    return df

def preprocess_data(df):
    df.fillna('', inplace=True)  # Fill missing values with empty strings for text data
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

def create_graph(df, features):
    G = nx.Graph()
    
    df['post_id'] = df.index  # Use index as 'post_id'

    for idx, row in df.iterrows():
        G.add_node(row['post_id'], features=features[idx])
    
    dense_features = features.toarray()
    
    for i in range(len(dense_features)):
        for j in range(i + 1, len(dense_features)):
            similarity = np.dot(dense_features[i], dense_features[j])
            if similarity > 0.5:
                G.add_edge(df.loc[i, 'post_id'], df.loc[j, 'post_id'])
    
    return G

def detect_communities(G):
    partition = community_louvain.best_partition(G)
    return partition

def plot_communities(G, partition):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 12))
    
    num_communities = len(set(partition.values()))
    cmap = plt.get_cmap('viridis', num_communities)
    
    nx.draw_networkx_nodes(G, pos, node_size=50, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    plt.title('Community Detection using Louvain Method')
    plt.show()

def detect_fake_news(file_path):
    df = load_dataset(file_path)
    df = preprocess_data(df)
    features = extract_features(df)
    G = create_graph(df, features)
    partition = detect_communities(G)
    plot_communities(G, partition)

    # Map communities to predictions
    community_labels = np.array([partition.get(post_id, -1) for post_id in df['post_id']])
    
    # Example mapping (this might need adjustment based on your specific needs)
    # For now, we assume that community 1 is fake news and community 0 is true news
    predictions = np.where(community_labels == 1, 1, 0)

    fake_news_df = df[predictions == 1]
    true_news_df = df[predictions == 0]
    fake_news_df.to_csv("fake_news.csv", index=False)
    true_news_df.to_csv("true_news.csv", index=False)
    print("\nFake news saved to 'fake_news.csv'")
    print("True news saved to 'true_news.csv'")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(predictions)), predictions, c=predictions, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.title('Distribution of Predicted Communities')
    plt.xlabel('Data Point')
    plt.ylabel('Predicted Community')
    plt.colorbar(label='Community')
    plt.show()
    
    return predictions, df['fake_status'].values

def evaluate_model(true_labels, predictions, class_label):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, pos_label=class_label, zero_division=0)
    recall = recall_score(true_labels, predictions, pos_label=class_label, zero_division=0)
    f1 = f1_score(true_labels, predictions, pos_label=class_label, zero_division=0)
    return accuracy, precision, recall, f1

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