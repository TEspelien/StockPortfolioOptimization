import pandas as pd
import yfinance as yf

# tickers =["XLC", "XLP", "XLY", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
tickers = ['NVDA', 'JPM', 'LLY', 'GE', 'GBTC']

# download weekly data for tickers
df = yf.download(tickers, interval='1wk', start='2023-01-01', end='2023-12-31', group_by='ticker')

# the download seems to change the order of tickers so resort before continuing
df = df.loc[:, tickers]

# each ticker has 6 columns, only keep 0 & 4
# drop 1, 2, 3, 5 for each ticker
column_idx_to_drop = [k for k in range(6 * len(tickers)) if k % 6 not in [0, 4]]
df = df.drop(df.columns[column_idx_to_drop], axis=1)  # df.columns is zero-based pd.Index

print(df.head())
df.to_csv('data/open_close.csv', index=False)

# calculate weekly returns
def calc_returns(row):
    returns = []

    for column_idx in range(len(tickers)):
        open_ = row[column_idx * 2]
        close = row[column_idx * 2 + 1]
        returns.append((close - open_) / open_)

    return returns

weekly_returns = pd.DataFrame(df.apply(calc_returns, axis=1).tolist(), columns=tickers)

print(weekly_returns.head())
weekly_returns.to_csv('data/weekly_returns.csv', index=False)
