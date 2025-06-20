# source myenv/bin/activate for the sentimentAnalysis directory

from dotenv import load_dotenv
import os
import praw
from textblob import TextBlob

# Load .env file
load_dotenv()

# Reddit API credentials
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

subreddit = reddit.subreddit('IsraelPalestine')
with open('reddit_posts.txt', 'w', encoding='utf-8') as f:
    for post in subreddit.search('Israel Iran', limit=100):
        # Write both title and selftext to file
        f.write(f"User: {post.author}\n")
        f.write(f"Title: {post.title}\n")
        f.write(f"Content: {post.selftext}\n")
        f.write("---\n")  # Separator for readability


# Using Vader to assign sentiment scores
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 0)

posts = []
with open('reddit_posts.txt', 'r', encoding='utf-8') as file:
    post = {}
    for line in file:
        if line.startswith('User:'):
            post['user'] = line[len('User:'):].strip()
        elif line.startswith('Title:'):
            post['title'] = line[len('Title:'):].strip()
        elif line.startswith('Content:'):
            post['content'] = line[len('Content:'):].strip()
        elif line.strip() == "---":
            if post:
                posts.append(post)
                post = {}

analyzer = SentimentIntensityAnalyzer()
for post in posts:
    sentiment = analyzer.polarity_scores(post['content'])
    post['sentiment'] = sentiment['compound']

df = pd.DataFrame(posts)
print(df[['user', 'title', 'sentiment']])
user_sentiment = df.groupby('user')['sentiment'].agg(['mean', 'std', 'count']).reset_index()
user_sentiment = user_sentiment.sort_values(by='mean', ascending=False)
print(user_sentiment) # got user sentiments

print(df[df['user'] == 'thatshirtman'])
print(df[df['user'] == 'soosoolaroo'])


# Cluster users based on sentiment
from sklearn.cluster import KMeans

X = user_sentiment[['mean']].values
kmeans = KMeans(n_clusters = 5, random_state=0)
kmeans.fit(X)
user_sentiment['cluster'] = kmeans.labels_
print(user_sentiment)

# Save the user sentiment data to a CSV file
user_sentiment.to_csv('user_sentiment.csv', index=False)


# Visualize the sentiment distribution
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(user_sentiment['mean'], bins=20, color='blue', alpha=0.7)
plt.title('Distribution of User Sentiment')
plt.xlabel('Sentiment Score')
plt.ylabel('Number of Users')
plt.axvline(user_sentiment['mean'].mean(), color='red', linestyle='dashed', linewidth=1)
