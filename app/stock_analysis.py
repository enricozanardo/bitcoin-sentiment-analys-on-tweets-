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

def prediction_data(stock, days):
    '''
    stock = 'BTC-USD'
    days = 5 days to predict in future
    start/end = historical dataset
    '''
    # start = datetime.datetime(2019, 8, 1)
    # end = datetime.datetime(2019, 9, 3)

    # get the data from internet
    # try:
    #     df = web.DataReader(stock, 'yahoo', start, end)
    #     print(df.tail())
    # except ():
    #     print('it eas not possible to get the data.')

    df = pd.read_csv('{}/BTC-USD_export.csv'.format(data_folder))
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
    df['prediction'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    forecast_days = int(days)

    #Predicting the stock price in the future
    # Random shuffle the dataset
    # df = df.sample(frac=1)

    # Set the features columns
    X = np.array(df.drop(['prediction', 'Date'], 1))
    # Set the target column
    Y = np.array(df['prediction'])
    # Standardize a dataset along any axis
    X = preprocessing.scale(X)
    # Split the dataset to 45% testing and then 55% training sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.45)

    # Performing the Regression on the trainig data
    linear_regression_classifier = LinearRegression()
    linear_regression_classifier.fit(X_train, Y_train)
    X_prediction = X[-forecast_days:]
    prediction_linear_regression = (linear_regression_classifier.predict(X_prediction))
    plr = round(float(np.float64(prediction_linear_regression[0])), 2)

    # Print out the Linear Regression prediction
    print('Prediction at {} days using linear regression is about {} $'.format(days, str(plr)))

    # quadratic, linear, lasso, ridge
    quadratic_regression_classifier = make_pipeline(PolynomialFeatures(2), Ridge())
    quadratic_regression_classifier.fit(X_train, Y_train)
    prediction_quadratic_regression = quadratic_regression_classifier.predict(X_prediction)
    pqr = round(float(np.float64(prediction_quadratic_regression[0])), 2)

    # Print out the Quadratic regression prediction
    print('Prediction at {} days using quadratic regression is about {} $'.format(days, str(pqr)))

    # KNN Regression
    kneighbor_regression_classifier = KNeighborsRegressor(n_neighbors=2)
    kneighbor_regression_classifier.fit(X_train, Y_train)
    prediction_kneighbor_regression = kneighbor_regression_classifier.predict(X_prediction)
    pkr = round(float(np.float64(prediction_kneighbor_regression[0])), 2)

    # Print out the Quadratic regression prediction
    print('Prediction at {} days using K Nearest Neighbor (KNN) regression is about {} $'.format(days, str(pkr)))

    ## Work on the tweets Dataset
    tweet_df = pd.read_csv('{}/analysis_all.csv'.format(data_folder))

    tweet_df['number'] = tweet_df['tweet'].shift()
    tweet_df.dropna(inplace=True)

    # group_by_date_sentiment = tweet_df.groupby(['created_at', 'sentiment'])['number'].agg('sum')
    group_by_date_sentiment = tweet_df.groupby(['created_at', 'sentiment'], as_index=False).count().pivot('created_at', 'sentiment').fillna(0)

    df_tmp = group_by_date_sentiment['number']
    # print(group_by_date_sentiment['number'].head())
    df_values = df.set_index('Date')
    # print(df_values.head())

    final_df = pd.merge(df_values, df_tmp, left_index=True, right_index=True)

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
    plt.savefig('sentiment.png')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    stock = 'BTC-USD'
    days = 1

    prediction_data(stock, days)

# start = datetime.datetime(2019, 1, 1)
# end = datetime.datetime(2019, 9, 3)
#
# df = web.DataReader("BTC-USD", 'yahoo', start, end)
# print(df.tail())
#
# # Determine trend using Moving Average and remove "noises".
# final_close_price = df['Adj Close']
# last_days = 20
# moving_avarage = final_close_price.rolling(window=last_days).mean()
# # print(moving_avarage)
#
# # Adjust the size of mathplotlib
# mpl.rc('figure', figsize=(8, 7))
# mpl.__version__
#
# # Adjust the style of matplotlib
# style.use('ggplot')
#
# tweet_df = pd.read_csv('analysis.csv')
# LE = LabelEncoder()
# tweet_df['code'] = LE.fit_transform(tweet_df['sentiment'])
# tweet_sentiment_code = tweet_df['code']
# tweet_date = tweet_df['created_at']
#
# print(tweet_df.tail())
# # tweet_sentiment_code.plot(label='Sentiment')
# final_close_price.plot(label='BTC-USD')
# moving_avarage.plot(label='moving average based on last {} days.'.format(last_days))
#
# # plt.legend()
# # plt.show()
#
# # Determine Risk and Return
# rets = final_close_price / final_close_price.shift(1) - 1
# rets.plot(label = 'return')