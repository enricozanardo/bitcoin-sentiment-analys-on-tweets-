import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import DataFrame


start = datetime.datetime(2019, 1, 1)
end = datetime.datetime(2019, 9, 3)

df = web.DataReader("BTC-USD", 'yahoo', start, end)
print(df.tail())

# Determine trend using Moving Average and remove "noises".

