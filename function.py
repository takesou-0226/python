import numpy as np

def mse(y_true, y_pred):
    return (sum((y_pred - y_true) ** 2)) / len(y_true)

def linear_predict(X, w, b):
    return X @ w + b

def linear_gradient(X, y, w, b):
    y_pred = linear_predict(X, w, b)
    error = y_pred - y
    dw = (1 / len(y)) * (X.T @ error)
    db = (1 / len(y)) * sum(error)
    return dw, db

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_predict(X, w, b):
    y_pred = linear_predict(X, w, b)
    y_prob = sigmoid(y_pred)
    ans = (y_prob >= 0.5).astype(int)
    return ans, y_prob

def ReLU(x):
    return np.maximum(0, x)

