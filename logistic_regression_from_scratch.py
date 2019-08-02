import os
import pathlib
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def init_weights(x):
    return np.random.uniform(size=x.shape[1] + 1)

def calculate_input(x, weights):
    return np.dot(x, weights[1:]) + weights[0]

def calculate_errors(y, y_pred):
    return y - y_pred

def calculate_gradient(x, errors):
    return np.concatenate([np.array([errors.sum()]), np.dot(x.T, errors)])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def update_weights(learning_rate, weights, gradient):
    return weights + learning_rate * gradient

def train(x, y, n_iter, learning_rate):
    weights = init_weights(x)
    for i in range(n_iter):
        y_pred = sigmoid(calculate_input(x, weights))
        errors = calculate_errors(y, y_pred)
        gradient = calculate_gradient(x, errors)
        weights = update_weights(learning_rate, weights, gradient)
    return weights

def predict(x, weights):
    return np.where(calculate_input(x, weights) > 0.0, 1, 0)

def save_model(weights):
    f = open('log_red_np_model_{}.pckl'.format(time.time()), 'wb')
    pickle.dumps(weights)
    f.flush()
    f.close()

def load_model(filename):
    f = open(model_name, 'rb')
    weights = pickle.loads(f)
    f.close()
    return weights
    
def main():
    X, y = make_moons()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    weights = train(X_train, y_train, 10000, 0.001)
    save_model(weights)
    y_pred = predict(X_test, weights)
    print("Accuracy: ", accuracy_score(y_test, y_pred))


#data = make_moons()
#print(type(data))
#print(data.shape)
#print(data[0].shape)

if __name__ == '__main__':
    main()

