import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

from sortedcontainers import SortedKeyList
from statistics import mean

import sys

mu_is = pd.read_csv('mu_is.csv').values

negative_mus = mu_is < 0
negative_mus = negative_mus.flatten()

variances = pd.read_csv('variances.csv').values

n = len(mu_is)

arguments = sys.argv[1:]

if len(arguments) > 0:
    mus = [float(arguments[0])]
else:
    mus = [m/10000.0 for m in range(50, 275, 25)]

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

def weekly_return_penalty(W, mu):
    d = weekly_return(W) - mu
    if d < 0:
        return 1e7 * d**2
    return 0

def allocation_penalty(W):
    s = sum(W)
    if s < 1:
        return 1e5 * (0.999-s)**2
    else:
        return 1e7 * (s-1)**2

def regularize(W):
    return sum([1e6 * w**2 for w in W if w < 0])

def custom_loss(W, mu):
    return risk_proxy(W) + regularize(W) + allocation_penalty(W) + weekly_return_penalty(W, mu)

p1_iter_counts = []

p1_num_attempts = 200
p1_loss_change_threshold = 1e-3 #stop once loss changes by less than 0.01%
p1_max_iterations = 250

rng = np.random.default_rng()

solutions = []

for mu in mus:

    #auto sorted list of portfolios, sorted by their loss
    #elements e: [[portfolio], loss]
    p1_minima = SortedKeyList(key = lambda e: e[1])
    p1_minima.add([[], 1e6])
    
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
        
        prev_loss = 1e7
        itr = 0

        # Optimization loop
        
        while itr < p1_max_iterations:

            with tf.GradientTape() as tape:
                loss = custom_loss(initial_guess, mu)
            
            gradients = tape.gradient(loss, [initial_guess])
            optimizer.apply_gradients(zip(gradients, [initial_guess]))

            loss = loss.numpy()

            loss_change = (loss - prev_loss) / prev_loss

            #print(itr, loss, loss_change)

            prev_loss = loss

            #early stopping: if loss is not good enough by a certain point, dont waste time on this attempt

            if itr == 75 and loss > 1000 * p1_minima[0][1]:
                break

            if itr == 125 and loss > 100 * p1_minima[0][1]:
                break

            if itr >= 180 and abs(loss_change) > p1_loss_change_threshold:
                break;

            itr += 1;


        p1_minima.add([initial_guess.numpy(), prev_loss])

        
        p1_iter_counts.append(itr)
        print("attempt ", attempt, "loss: ", prev_loss, "itr: ", itr)

    #focus on the best 10% of local minima found in phase 1

    #p1_minima = p1_minima[:int(p1_num_attempts * 0.1)]

    #print([e[1] for e in p1_minima])

    #p1_minima_df = pd.DataFrame(data = [np.append(e[0], e[1]) for e in p1_minima], columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'loss'])

    #p1_minima_df.to_csv('p1_minima.csv', index = False)

    optimal = p1_minima[0][0]

    solution = [mu*100] + [100 * x for x in optimal]

    print(solution)

    solutions.append(solution)


    print("Optimal solution:", optimal * 100)
    print("Total allocation:", sum(optimal) * 100)
    print("Final loss:", p1_minima[0][1])
    print("Risk proxy:", risk_proxy(optimal))
    print("Regularization penalty:", regularize(optimal))
    print("Allocation penalty:", allocation_penalty(optimal))
    print("Weekly return penalty:", weekly_return_penalty(optimal, mu))
    print("Average weekly returns:", weekly_return(optimal))
    print("Requested weekly return:", mu)


for s in solutions:
    print(s)

solutions_df = pd.DataFrame(data = solutions, columns = ['mu', 'x1', 'x2', 'x3', 'x4', 'x5'])
solutions_df.to_csv('solutions.csv', index = False)
