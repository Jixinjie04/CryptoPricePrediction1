import configparser
import re
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import pandas as pd
import numpy as np
import datetime
import time

config = configparser.ConfigParser()
config.read('config.ini')
api_key = config['binance']['api_key']
api_secret = config['binance']['api_secret']
client = Client(api_key, api_secret, tld='us')
tickers = client.get_all_tickers()
tickers_df = pd.DataFrame(tickers)
tickers_df['price'] = tickers_df['price'].astype(float)
tickers_df.set_index('symbol', inplace=True)
print(tickers_df)

symbol = "BTCUSDT"
interval = "12h"
klines = client.get_historical_klines(symbol, interval, "1 Jan, 2020", "1 Jan, 2025")
klines_df = pd.DataFrame(klines, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                          'Close Time', 'Quote Asset Volume', 'Number of Trades',
                                          'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume',
                                          'Ignore'])

print(klines_df)

tickers_df.to_csv('tickers.csv')
klines_df.to_csv('klines.csv')
print(klines_df.dtypes)