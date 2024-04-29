import numpy as np
import pandas as pd
import math

from matplotlib import pyplot as plt

data = pd.read_csv('solutions.csv')

tickers = ["NVDA", "JPM", "LLY", "GE", "GBTC"]

ax = data.plot.bar(x='mu', stacked=True,
        title='Stacked Bar Graph by dataframe')

# Add labels and legend
ax.set_xlabel('Weekly Returns (%)')
ax.set_ylabel('Portfolio Composition')
ax.set_title('Portfolios for Increasing Weekly Returns')
ax.legend(tickers,
          title='Components', bbox_to_anchor=(0.9, 1), loc='upper left')

# Show the plot
plt.show()
