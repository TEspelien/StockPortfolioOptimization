import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

from statistics import mean

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

# function to optimize:
# sigma_sq(W) with penalties
# regularize weights
# sum(W) as close to 1
# weekly returns as close to target

def risk_proxy(W):
    return 1e5 * sigma_sq(W)

def regularize(W):
    return 1e-1 * sum(W**2)

def allocation_penalty(W):
    return 1e1 * (1 - sum(W))**2

def weekly_return_penalty(W):
    return 1e6 * (weekly_return(W) - mu)**2

def custom_loss(W):
    return risk_proxy(W) + regularize(W) + allocation_penalty(W) + weekly_return_penalty(W)

#pick many different initial points and optimize from each, then pick the best

minima = []

iter_counts = []

num_attempts = 10
loss_change_threshold = 0.005 #stop once loss changes by less than 0.5%
max_num_iterations = 100

rng = np.random.default_rng()

for attempt in range(num_attempts):

    #start by randomly setting n-1 weights
    #normal distribution parameter picker: https://www.desmos.com/calculator/jxzs8fz9qr 
    initial_guess = rng.normal(loc = 0, scale = 0.15, size = n-1)

    #change which weight is calculated last because it tends to have the highest magnitude

    #calculate the value of the last weight in order to satisfy W_i * mu_i = mu
    #w_n * mu_n + rest = mu
    w_n = (mu - initial_guess.dot(mu_is[:-1]))/mu_is[-1]

    initial_guess = np.append(initial_guess, w_n)

    print(sum(initial_guess), weekly_return(initial_guess))
    
    initial_guess = tf.Variable(initial_value = initial_guess, trainable = True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # Optimization loop

    prev_loss = 1e6
    loss_change = -1.0
    itr = 0

    #note that loss_change is negative
    #stop optimizing once loss is only improving a little bit or enough iterations have passed
    while(-loss_change > loss_change_threshold and itr < max_num_iterations):

        with tf.GradientTape() as tape:
            loss = custom_loss(initial_guess)
        
        gradients = tape.gradient(loss, [initial_guess])
        optimizer.apply_gradients(zip(gradients, [initial_guess]))

        loss = loss.numpy()[0]

        loss_change = (loss-prev_loss) / prev_loss

        #print(itr, loss, loss_change)

        prev_loss = loss
        itr += 1

    minima.append([initial_guess.numpy(), custom_loss(initial_guess.numpy())])
    iter_counts.append(itr)
    print(attempt, minima[-1][1], itr)

abs_min = minima[0][0]
minimized = custom_loss(abs_min)

for m in minima:
    if(m[1] < minimized):
       abs_min = m[0]
       minimized = m[1]


print("Optimal solution:", abs_min * 100)
print("Total allocation:", sum(abs_min) * 100)
print("Final loss:", minimized[0])
print("Risk proxy:", risk_proxy(abs_min))
print("Regularization penalty:", regularize(abs_min))
print("Allocation penalty:", allocation_penalty(abs_min))
print("Weekly return penalty:", weekly_return_penalty(abs_min))
print("Average weekly returns:", weekly_return(abs_min)[0])
print("Requested weekly return:", mu)
print("Mean iterations taken per attempt:", mean(iter_counts))
print("Attempts with max iterations taken:", iter_counts.count(max_num_iterations))
