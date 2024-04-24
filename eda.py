import pandas as pd
import numpy as np
import math

weekly_returns = pd.read_csv('data/weekly_returns.csv')

tickers = weekly_returns.columns

#find the mean of each column of weekly returns
mu_is = pd.DataFrame(weekly_returns.apply(lambda col: col.mean()))

#calculate the covariance of each pair i, j of tickers
#note that cov(i, i) is simply its variance
variances = pd.DataFrame([[weekly_returns.iloc[:, i].cov(weekly_returns.iloc[:, j]) \
             for j in range(len(tickers))] for i in range(len(tickers))])

print(mu_is)

variances.style.format("{:.5f}")

mu_is.to_csv('data/mu_is.csv', index = False)

variances.to_csv('data/variances.csv', index = False)
