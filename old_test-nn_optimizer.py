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
    mu = arguments[0]
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
    return weekly_return(W) - mu

# Generate training data including constraints
np.random.seed(0)

# randomly generate raw weights and normalize to sum to 1
x_train = [raw/sum(raw) for raw in np.random.uniform(0, 1, (100, n))]

# the function we are trying to optimize is sigma_sq(W) with a penalty given by constraint(W, mu)

def to_minimize(W):
    return sigma_sq(W) + 10 * constraint(W)**2

y_train = [to_minimize(W) for W in x_train]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

# Create a neural network model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(n,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output the function value
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(train_dataset, epochs=50)

initial_guess = tf.convert_to_tensor([1/n for _ in range(n)])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

#@tf.function
def optimize():
    with tf.GradientTape() as tape:
        tape.watch(initial_guess)
        y_pred = model(tf.expand_dims(initial_guess, axis=0))
        loss = tf.reduce_mean(tf.square(y - y_pred))
        loss = tf.squeeze(y_pred)
    gradients = tape.gradient(loss, initial_guess)
    print(gradients, initial_guess)
    optimizer.apply_gradients([(gradients, initial_guess)])
    return initial_guess

for _ in range(100):
    optimal_solution = optimize()

print("Optimal solution:", optimal_solution.numpy())
print("Minimum sigma_sq:", sigma_sq(optimal_solution))
print("Average weekly returns:", weekly_return(optimal_solution))
