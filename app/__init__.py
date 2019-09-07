from flask import Flask, render_template, request, make_response
from dotenv import load_dotenv
from app import stock_analysis
import json
import time
load_dotenv()
import os.path
from os import path
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    user = "Enrico"
    return render_template('index.html', name=user)

@app.route('/ic')
def image_classifier():
    return render_template('image_classifier.html')

@app.route('/st')
def stock_prediction():

    #TODO: Get the list of positive, neutral and negative tweets
    print(path.exists('app/exports/analysis_all.csv'))
    tweet_df = pd.read_csv('app/exports/analysis_all.csv')

    tweet_df.drop('remove', axis=1, inplace=True)
    tweet_df.dropna(inplace=True)

    print(tweet_df.tail())

    positive_tweets = tweet_df[tweet_df['sentiment'] == 'Positive']
    positive_tweets_text = positive_tweets['tweet']
    positive_tweets_text = positive_tweets_text.sample(frac=1)
    first_ten_positive = positive_tweets_text.head(10)

    positive =  first_ten_positive.values.tolist()

    neutral_tweets = tweet_df[tweet_df['sentiment'] == 'Neutral']
    neutral_tweets_text = neutral_tweets['tweet']
    neutral_tweets_text = neutral_tweets_text.sample(frac=1)
    first_ten_neutral = neutral_tweets_text.head(10)

    neutral = first_ten_neutral.values.tolist()

    negative_tweets = tweet_df[tweet_df['sentiment'] == 'Negative']
    negative_tweets_text = negative_tweets['tweet']
    negative_tweets_text = negative_tweets_text.sample(frac=1)
    first_ten_negative = negative_tweets_text.head(10)

    negative = first_ten_negative.values.tolist()


    #
    # # group_by_date_sentiment = tweet_df.groupby(['created_at', 'sentiment'])['number'].agg('sum')
    # group_by_date_sentiment = tweet_df.groupby(['created_at', 'sentiment'], as_index=False).count().pivot('created_at',
    #                                                                                                       'sentiment').fillna(
    #     0)

    return render_template('stock_prediction.html', positive_tweets=positive, neutral_tweets=neutral, negative_tweets=negative)


@app.route('/predict', methods=['POST'])
def predict():
    days = request.form['days'];
    stock = "BTC-USD"

    # TODO: check if days are numbers!
    maxConfidenceItem =  stock_analysis.prediction_data(stock, days)
    # maxConfidenceItem = stock_analysis.fake_predict(stock, days)
    # time.sleep(2)
    # print(maxConfidenceItem)
    maxConfidenceItem['prediction'] = 'Prediction:  {} $'.format(str(maxConfidenceItem['prediction']))
    maxConfidenceItem['confidence'] = 'Confidence: {}%'.format(str(maxConfidenceItem['confidence']))

    return json.dumps({'status':'OK','result':maxConfidenceItem});

