import pandas as pd

# Load the dataset
df = pd.read_csv("facebook.csv")

# Calculate total news
total_news = df.shape[0]

# Calculate total likes, shares, comments
total_likes = df['likes'].sum()
total_shares = df['shares'].sum()
total_comments = df['comments'].sum()

# Calculate total words and average word length
total_words = df['words'].sum()
total_avg_word_length = df['avg_word_length'].mean()

# Calculate total marks and exclamation marks
total_marks = df['marks'].sum()
total_exclamation_marks = df['exclamation_marks'].sum()

print("Total News:", total_news)
print("Total Likes:", total_likes)
print("Total Shares:", total_shares)
print("Total Comments:", total_comments)
print("Total Words:", total_words)
print("Average Word Length:", total_avg_word_length)
print("Total Marks:", total_marks)
print("Total Exclamation Marks:", total_exclamation_marks)
