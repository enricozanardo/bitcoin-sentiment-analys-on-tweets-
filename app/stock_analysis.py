import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
from scipy.ndimage.filters import gaussian_filter1d
import os.path
from os import path


# For Prediction
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor


data_folder = 'exports'


def fake_predict(stock, days):
    return 'foo'

def prediction_data(stock, days):
    '''
    stock = 'BTC-USD'
    days = 5 days to predict in future
    start/end = historical dataset
    '''
    start = datetime.datetime(2019, 8, 1)
    end = datetime.datetime(2019, 9, 6)

    # Store in an array the results of the three different classifier
    prediction_values = []

    # try to get the data from internet
    # try:
    #     stock_df = web.DataReader(stock, 'yahoo', start, end)
    #     # print(stock_df.tail())
    #     csv_name = ('{}/{}_export.csv'.format(data_folder, stock))
    #     stock_df.to_csv(csv_name)
    #
    # except ():
    #     print('it eas not possible to get the data.')

    print(path.exists('app/exports/BTC-USD_export.csv'))

    stock_df = pd.read_csv('app/exports/BTC-USD_export.csv')

    # print(df.tail())

    # save the data locally into a csv file
    # if os.path.exists('./{}'.format(data_folder)):
    #     pass
    # else:
    #     os.mkdir(data_folder)
    #
    # csv_name = ('{}/{}_export.csv'.format(data_folder, stock))
    # df.to_csv(csv_name)

    # add a column prediction to the dataset
    stock_df['prediction'] = stock_df['Close'].shift(-1)
    stock_df.dropna(inplace=True)

    # print(stock_df.tail())

    forecast_days = int(days)



    #Predicting the stock price in the future
    # Random shuffle the dataset
    # df = df.sample(frac=1)

    # Set the features columns
    X = np.array(stock_df.drop(['prediction', 'Date'], 1))
    # Set the target column
    Y = np.array(stock_df['prediction'])
    # Standardize a dataset along any axis
    X = preprocessing.scale(X)
    # Split the dataset to 45% testing and then 55% training sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.45)

    # Performing the Regression on the trainig data
    linear_regression_classifier = LinearRegression()
    linear_regression_classifier.fit(X_train, Y_train)
    X_prediction = X[-forecast_days:]
    prediction_linear_regression = (linear_regression_classifier.predict(X_prediction))
    confidence_lr = linear_regression_classifier.score(X_train, Y_train)
    plr = round(float(np.float64(prediction_linear_regression[0])), 2)
    clr = round(float(np.float64(confidence_lr*100)), 2)

    linear_regression_prediction = {}
    linear_regression_prediction['prediction'] = plr
    linear_regression_prediction['confidence'] = clr
    # Add to the array the results
    prediction_values.append(linear_regression_prediction)

    # Print out the Linear Regression prediction
    print('Prediction at {} days using linear regression is about {} $'.format(days, str(plr)))
    print('Confidence at {} days using linear regression is about {}% '.format(days, str(clr)))

    # quadratic, linear, lasso, ridge
    quadratic_regression_classifier = make_pipeline(PolynomialFeatures(2), Ridge())
    quadratic_regression_classifier.fit(X_train, Y_train)
    prediction_quadratic_regression = quadratic_regression_classifier.predict(X_prediction)
    confidence_pq = quadratic_regression_classifier.score(X_train, Y_train)
    pqr = round(float(np.float64(prediction_quadratic_regression[0])), 2)
    cpq = round(float(np.float64(confidence_pq * 100)), 2)

    quadratic_regression_prediction = {}
    quadratic_regression_prediction['prediction'] = pqr
    quadratic_regression_prediction['confidence'] = cpq
    # Add to the array the results
    prediction_values.append(quadratic_regression_prediction)

    # Print out the Quadratic regression prediction
    print('Prediction at {} days using quadratic regression is about {} $'.format(days, str(pqr)))
    print('Confidence at {} days using quadratic regression is about {}%'.format(days, str(cpq)))

    # KNN Regression
    kneighbor_regression_classifier = KNeighborsRegressor(n_neighbors=2)
    kneighbor_regression_classifier.fit(X_train, Y_train)
    prediction_kneighbor_regression = kneighbor_regression_classifier.predict(X_prediction)
    confidence_kr = kneighbor_regression_classifier.score(X_train, Y_train)
    pkr = round(float(np.float64(prediction_kneighbor_regression[0])), 2)
    ckr = round(float(np.float64(confidence_kr * 100)), 2)

    kneighbor_regression_prediction = {}
    kneighbor_regression_prediction['prediction'] = pkr
    kneighbor_regression_prediction['confidence'] = ckr
    # Add to the array the results
    prediction_values.append(kneighbor_regression_prediction)

    # Print out the Quadratic regression prediction
    print('Prediction at {} days using K Nearest Neighbor (KNN) regression is about {} $'.format(days, str(pkr)))
    print('Confidence at {} days using K Nearest Neighbor (KNN) regression is about {}%'.format(days, str(ckr)))

    ## Work on the tweets Dataset
    print(path.exists('app/exports/analysis_all.csv'))
    tweet_df = pd.read_csv('app/exports/analysis_all.csv')

    tweet_df['number'] = tweet_df['tweet'].shift()
    tweet_df.dropna(inplace=True)

    # group_by_date_sentiment = tweet_df.groupby(['created_at', 'sentiment'])['number'].agg('sum')
    group_by_date_sentiment = tweet_df.groupby(['created_at', 'sentiment'], as_index=False).count().pivot('created_at', 'sentiment').fillna(0)

    # print(group_by_date_sentiment)

    df_tmp = group_by_date_sentiment['number']
    # print(group_by_date_sentiment['number'].head())
    df_values = stock_df.set_index('Date')

    final_df = pd.merge(df_values, df_tmp, left_index=True, right_index=True)

    # print(final_df)

    # Work with graph
    columns_df = final_df[['Close', 'Neutral', 'Positive']]
    x = columns_df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    plot_df = pd.DataFrame(x_scaled, columns=columns_df.columns, index=columns_df.index)

    final_close_price = plot_df['Close']
    neutral = plot_df['Neutral']
    positive = plot_df['Positive']

    # Print the graph
    # Adjust the size of mathplotlib
    mpl.rc('figure', figsize=(8, 7))
    mpl.__version__

    plt.suptitle('Bitcoin Sentiment Analysis on Tweets', fontsize=14, fontweight='bold')
    plt.ylabel('Sentiment')
    plt.xlabel('Time')
    # Adjust the style of matplotlib
    style.use(['ggplot'])
    style.context('Solarize_Light2')

    neutral.plot(
        label='Neutral Tweets',
        color='orange',
        linestyle='dashed',
        linewidth=2,
        alpha=0.5,
        marker='s',
        markersize=5,
        markerfacecolor='blue',
        markeredgecolor='blue',
    )
    positive.plot(
        color='green',
        linestyle='dashed',
        linewidth=2,
        alpha=0.5,
        marker='*',
        markersize=5,
        markerfacecolor='blue',
        markeredgecolor='blue',
        label='Positive Tweets'
    )
    final_close_price.plot(
        color='red',
        linestyle='solid',
        linewidth=4,
        alpha=0.5,
        marker='o',
        markersize=5,
        markerfacecolor='blue',
        markeredgecolor='blue',
        label='BTC-USD'
    )
    plt.legend()

    #save to file
    plt.savefig('app/static/img/sentiment.png')
    # plt.show()
    plt.close()

    # return the price with the best confidence
    maxConfidenceItem = max(prediction_values, key=lambda x: x['confidence'])

    print('maxConfidenceItem: {}'.format(str(maxConfidenceItem)))

    return maxConfidenceItem

if __name__ == '__main__':
    stock = 'BTC-USD'
    days = 5

    foo = prediction_data(stock, days)
    print('foo {}'.format(str(foo)))