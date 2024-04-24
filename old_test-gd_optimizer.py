import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

import sys

mu_is = pd.read_csv('data/mu_is.csv').values

variances = pd.read_csv('data/variances.csv').values

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
    return 100 * (weekly_return(W) - mu)**2

# the function we are trying to optimize is sigma_sq(W) with a penalty given by constraint(W, mu)

def to_minimize(W):
    W = W * 1/sum(W)
    return sigma_sq(W) + constraint(W)

initial_guess = tf.Variable(initial_value = np.array([1/n for _ in range(n)]), trainable=True) 

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)


# Optimization loop
num_iterations = 100
for iter in range(num_iterations):
    with tf.GradientTape() as tape:
        loss = to_minimize(initial_guess)
    
    gradients = tape.gradient(loss, [initial_guess])
    optimizer.apply_gradients(zip(gradients, [initial_guess]))

    #print progress every 10%
    if iter % (num_iterations/10) == 0:
        print(loss)
        print(initial_guess)

print("Optimal solution:", initial_guess.numpy() * 100)
print("Total allocation:", sum(initial_guess.numpy()) * 100)
print("Minimum sigma_sq:", sigma_sq(initial_guess).numpy())
print("Final constraint:", constraint(initial_guess)[0])
print("Average weekly returns:", weekly_return(initial_guess)[0])
