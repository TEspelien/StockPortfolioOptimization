import pandas as pd
import numpy as np
import math
import yfinance as yf

tickers =["NVDA", "JPM", "LLY", "GE", "GBTC"]

df = yf.download(tickers, interval = '1wk', start = '2023-01-01', end = '2023-12-31', group_by = 'ticker')

#the download seems to change the order of tickers so resort before continuing
sorted_columns = [t for t in tickers if t in df.columns]

df = df.loc[:,sorted_columns]

#each ticker has 6 columns, only keep 0 & 4
#drop 1, 2, 3, 5 for each ticker

to_drop = [k for k in range(6 * len(tickers)) if k%6 not in [0, 4]]

df = df.drop(df.columns[to_drop], axis=1)  # df.columns is zero-based pd.Index

print(df.head())

df.to_csv('open_close.csv', index=False)


#calculate weekly returns

def calc_returns(row):

    returns = []

    for ticker_idx in range(len(tickers)):
        open_ = row[ticker_idx * 2]
        close = row[ticker_idx * 2 + 1]
        returns.append((close - open_) / open_)

    return returns

weekly_returns = pd.DataFrame(df.apply(calc_returns, axis=1).tolist(), columns=tickers)

print(weekly_returns.head())

weekly_returns.to_csv('weekly_returns.csv', index = False)



