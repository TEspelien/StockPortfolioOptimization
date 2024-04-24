import pandas as pd

weekly_returns = pd.read_csv('data/weekly_returns.csv')

tickers = weekly_returns.columns

# find the mean of each column of weekly returns
mu_vec = pd.DataFrame.mean(weekly_returns)
print(mu_vec)

# calculate the covariance of each pair of tickers
covariance_mtx = pd.DataFrame.cov(weekly_returns)
print(covariance_mtx)

# save the mean and covariance matrix to csv
mu_vec.to_csv('data/mu_vec.csv')
covariance_mtx.to_csv('data/covariance_mtx.csv')
