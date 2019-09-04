import tweepy
import re
import os
from textblob import TextBlob
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

consumer_key = os.getenv("CONSUMER_KEY")
consumer_secret = os.getenv("CONSUMER_SECRET_KEY")

access_token_key = os.getenv("ACCESS_TOKEN_KEY")
access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)
# api = tweepy.API(auth)
topicname = os.getenv("TOPIC_NAME", "enrico zanardo")
public_tweets = tweepy.Cursor(api.search, q=topicname, count=10, lang="en", since="2019-01-01").items()
unwanted_words = ['@', 'RT', ':', 'https', 'http']
symbols = ['@', '#']
data = []

for tweet in public_tweets:
    text = TextBlob(tweet.text.encode('utf-8').lower())
    text.correct()
    text_words = text.split()
    # print(text_words)
    cleaned_tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(RT)", " ", text).split())
    print(cleaned_tweet)
    print(TextBlob(cleaned_tweet).tags)
    analysis = TextBlob(cleaned_tweet)
    print(analysis.sentiment)
    polarity = 'Positive'
    if (analysis.sentiment.polarity < 0):
        polarity = 'Negative'
    if (analysis.sentiment.polarity <= 0.2):
        polarity = 'Neutral'
    print(polarity)
    dic={}
    dic['created_at'] = tweet.created_at
    dic['sentment'] = polarity
    dic['tweet'] = cleaned_tweet
    data.append(dic)


df = pd.DataFrame(data)
df.to_csv('analysis.csv')