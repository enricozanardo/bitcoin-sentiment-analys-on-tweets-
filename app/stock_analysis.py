import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl


start = datetime.datetime(2019, 1, 1)
end = datetime.datetime(2019, 9, 3)

df = web.DataReader("BTC-USD", 'yahoo', start, end)
print(df.tail())


# Determine trend using Moving Average and remove "noises".
final_close_price = df['Adj Close']
last_days = 150
moving_avarage = final_close_price.rolling(window=last_days).mean()
# print(moving_avarage)

# Adjust the size of mathplotlib
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjust the style of matplotlib
style.use('ggplot')

final_close_price.plot(label='BTC-USD')
moving_avarage.plot(label='moving average')
plt.legend()
