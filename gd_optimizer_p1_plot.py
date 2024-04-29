import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

import sys

from matplotlib import pyplot as plt

mu_is = pd.read_csv('mu_is.csv').values

negative_mus = mu_is < 0
negative_mus = negative_mus.flatten()

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
    return 1e3 * sigma_sq(W)

def weekly_return_penalty(W):
    return 1e7 * (weekly_return(W) - mu)**2

def allocation_penalty(W):
    s = sum(W)
    if s < 1:
        return 1e5 * (0.999-s)**2
    else:
        return 1e6 * (s-1)**2

def regularize(W):
    return sum([1e5 * w**2 for w in W if w < 0])

def custom_loss(W):
    return risk_proxy(W) + regularize(W) + allocation_penalty(W) + weekly_return_penalty(W)


p1_num_attempts = 10
p1_loss_change_threshold = 1e-3 #stop once loss changes by less than 0.1%
p1_max_iterations = 200

best_loss = 1e6

rng = np.random.default_rng()

curves = []

#phase 1:

#pick many different initial points and optimize from each,
#then pick the best local minima to explore further in phase 2
for attempt in range(p1_num_attempts):

    #start by randomly setting n-1 weights
    #normal distribution parameter picker: https://www.desmos.com/calculator/jxzs8fz9qr 
    initial_guess = rng.normal(loc = 0.2, scale = 0.15, size = n-1)

    #calculate the value of the last weight in order to satisfy W_i * mu_i = mu
    #w_n * mu_n + rest = mu

    w_n = (mu - initial_guess.dot(mu_is[:-1]))/mu_is[-1]

    initial_guess = np.append(initial_guess, w_n)

    #for the bonus, shorting is not allowed so if the mu < 0 we don't want that stock
    initial_guess[negative_mus] = 0

    initial_guess = tf.Variable(initial_value = initial_guess, trainable = True)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.00005, momentum = 0.8, clipvalue = 1.0)
    # Optimization loop

    prev_loss = 1e6
    loss_change = -1.0
    itr = 0

    curve = []

    #note that loss_change is negative

    #stopping conditions:
    #loss improves by less than a given threshold
    #max number of iterations have passed
    
    while itr < p1_max_iterations:

        with tf.GradientTape() as tape:
            loss = custom_loss(initial_guess)
        
        gradients = tape.gradient(loss, [initial_guess])
        optimizer.apply_gradients(zip(gradients, [initial_guess]))

        loss = loss.numpy()[0]

        loss_change = (loss - prev_loss) / prev_loss

        #print(itr, loss, loss_change)

        prev_loss = loss
        itr += 1

        curve.append(loss)

        #early stopping: if loss is not good enough by a certain point, dont waste time on this attempt

        if itr == 60 and loss > 1000 * best_loss:
            break

        if itr == 120 and loss > 100 * best_loss:
            break

        if itr >= 180 and abs(loss_change) > p1_loss_change_threshold:
            break;

    if(prev_loss < best_loss):
        best_loss = prev_loss

    curves.append(curve)

    print("attempt ", attempt, "loss: ", prev_loss, "itr: ", itr)


for curve in curves:
    x = range(1, len(curve) + 1)  # X values are indices of the points
    plt.plot(x, curve)

# Add labels and title
plt.xlabel('Iteration')
plt.ylabel('Loss (log scale)')
plt.title('Iteration - Loss Curves')

plt.yscale('log')

# Add a legend
#plt.legend(['Curve 1', 'Curve 2', 'Curve 3'])

plt.show()
