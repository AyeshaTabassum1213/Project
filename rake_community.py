import pandas as pd
from rake_nltk import Rake
import networkx as nx
from networkx.algorithms import community
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

def build_graph(df):
    G = nx.Graph()
    # Add nodes with RAKE keywords as attributes
    for idx, row in df.iterrows():
        keywords = extract_keywords_rake(row['post_text'])
        G.add_node(idx, keywords=keywords)
    
    # Add edges based on similarity of keywords
    for i in G.nodes:
        for j in G.nodes:
            if i < j:
                # Calculate similarity based on keyword overlap
                keywords_i = set(extract_keywords_rake(df.iloc[i]['post_text']).split())
                keywords_j = set(extract_keywords_rake(df.iloc[j]['post_text']).split())
                similarity = len(keywords_i & keywords_j) / len(keywords_i | keywords_j)
                if similarity > 0.2:  # Threshold for adding edges
                    G.add_edge(i, j, weight=similarity)
    return G

def detect_communities(G):
    communities = community.greedy_modularity_communities(G)
    community_dict = {}
    for i, com in enumerate(communities):
        for node in com:
            community_dict[node] = i
    return community_dict

def evaluate_communities(df, community_dict):
    true_labels = df['fake_status']
    predicted_labels = [community_dict.get(i, -1) for i in range(len(df))]
    
    # For evaluation, we need to map community labels to true labels
    community_labels = {}
    for i in set(predicted_labels):
        mode_label = pd.Series([true_labels[j] for j in range(len(true_labels)) if predicted_labels[j] == i]).mode()
        community_labels[i] = mode_label.iloc[0] if not mode_label.empty else -1
    
    mapped_labels = [community_labels.get(label, -1) for label in predicted_labels]
    
    accuracy = accuracy_score(true_labels, mapped_labels)
    precision = precision_score(true_labels, mapped_labels, pos_label=1, zero_division=0)
    recall = recall_score(true_labels, mapped_labels, pos_label=1, zero_division=0)
    f1 = f1_score(true_labels, mapped_labels, pos_label=1, zero_division=0)
    
    return accuracy, precision, recall, f1

def main():
    file_path = "facebook.csv"
    df = load_dataset(file_path)
    df = preprocess_data(df)
    G = build_graph(df)
    community_dict = detect_communities(G)
    accuracy, precision, recall, f1 = evaluate_communities(df, community_dict)
    
    print("Community Detection Performance Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    
    # Save the community detection results
    df['Community'] = [community_dict.get(i, -1) for i in range(len(df))]
    df.to_csv("community_detection_results.csv", index=False)
    print("Community detection results saved to 'community_detection_results.csv'")

if __name__ == "__main__":
    main()
