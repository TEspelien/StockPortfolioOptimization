import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

import sys

mu_is = pd.read_csv('mu_is.csv').values

variances = pd.read_csv('variances.csv').values

n = len(mu_is)

arguments = sys.argv[1:]

if len(arguments) > 0:
    mu = float(arguments[0])
else:
    mu = 0.01

# the variance of the function given weights W and variances previously calculated
def sigma_sq(W):
    return sum(W[i] * W[j] * variances[i, j] for i in range(n) for j in range(n))

def weekly_return(W):
    return np.dot(W, mu_is)

# define constraints such that the quantity is as close to 0
def constraint(W):

    # W * mu_is = mu
    return 

# function to optimize:
# sigma_sq(W) with penalties
# regularize weights
# sum(W) as close to 1
# weekly returns as close to target

def custom_loss(W):
    return sigma_sq(W) + sum(W**2) + 10 * (1 - sum(W))**2 + 100 * (weekly_return(W) - mu)**2

#pick many different initial points and optimize from each, then pick the best

minima = []

num_attempts = 20
num_iterations = 50

rng = np.random.default_rng()

for attempt in range(num_attempts):

    #start by randomly setting n-1 weights
    #normal distribution parameter picker: https://www.desmos.com/calculator/jxzs8fz9qr 
    initial_guess = rng.normal(loc = 0, scale = 0.2, size = n-1)

    #change which weight is calculated last because it tends to have the highest magnitude

    #calculate the value of the last weight in order to satisfy W_i * mu_i = mu
    #w_n * mu_n + rest = mu
    w_n = (mu - initial_guess.dot(mu_is[:-1]))/mu_is[-1]

    initial_guess = np.append(initial_guess, w_n)

    initial_guess = initial_guess / sum(initial_guess)
    
    initial_guess = tf.Variable(initial_value = initial_guess, trainable = True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    # Optimization loop
    for _ in range(num_iterations):
        with tf.GradientTape() as tape:
            loss = custom_loss(initial_guess)
        
        gradients = tape.gradient(loss, [initial_guess])
        optimizer.apply_gradients(zip(gradients, [initial_guess]))

    minima.append([initial_guess.numpy(), custom_loss(initial_guess.numpy())])
    print(attempt, minima[-1])

abs_min = minima[0][0]
minimized = custom_loss(abs_min)

for m in minima:
    if(m[1] < minimized):
       abs_min = m[0]
       minimized = m[1]


print("Optimal solution:", abs_min * 100)
print("Total allocation:", sum(abs_min) * 100)
print("Final loss:", minimized[0])
print("Average weekly returns:", weekly_return(abs_min)[0])
