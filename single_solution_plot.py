import numpy as np
import pandas as pd
import math

from matplotlib import pyplot as plt

solutions = pd.read_csv('solutions.csv').values

tickers =["NVDA", "JPM", "LLY", "GE", "GBTC"]

# Plot the bar graph
fig, ax = plt.subplots()

for i, solution in enumerate(solutions[:,1:5]):
    ax.bar(i, solution, bottom=np.sum(solutions[:i], axis=0), label=f'{solution[0] * 100}%')

# Add labels and title
ax.set_xlabel('Target Weekly Return')
ax.set_ylabel('Percent of Portfolio')
ax.set_title('Bar Graph with 5 Components Summing to 1')
ax.legend()

# Show the plot
plt.show()
