import numpy as np
import pandas as pd
import math

from matplotlib import pyplot as plt

data = pd.read_csv('solutions.csv')

variances = pd.read_csv('variances.csv').values
n = len(variances)

print(n)

tickers = ["NVDA", "JPM", "LLY", "GE", "GBTC"]

def sigma_sq(W):
    return sum(W[i] * W[j] * variances[i, j] for i in range(n) for j in range(n))

data['risk'] = data.apply(lambda row: sigma_sq(row[1:6]), axis = 1)

print(data)

ax = data.plot.line(x='mu', y='risk',
        title='mu - risk graph')

# Add labels and legend
ax.set_xlabel('Weekly Returns (%)')
ax.set_ylabel('Risk')
ax.set_title('Portfolio Risk for Increasing Weekly Returns')

# Show the plot
plt.show()
