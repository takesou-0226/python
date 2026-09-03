import numpy as np

# 平均二乗誤差
def mse(y_true, y_pred):
    return (sum((y_pred - y_true) ** 2)) / len(y_true)

# 線形回帰
def linear_predict(X, w, b):
    return X @ w + b

# 線形回帰の勾配
def linear_gradient(X, y, w, b):
    y_pred = linear_predict(X, w, b)
    error = y_pred - y
    dw = (1 / len(y)) * (X.T @ error)
    db = (1 / len(y)) * sum(error)
    return dw, db

# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ロジスティック回帰
def logistic_predict(X, w, b):
    y_pred = linear_predict(X, w, b)
    y_prob = sigmoid(y_pred)
    ans = (y_prob >= 0.5).astype(int)
    return ans, y_prob

# ReLU関数
def ReLU(x):
    return np.maximum(0, x)

# ソフトマックス関数
def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / sum(np.exp(x))

# 全結合層
def dense_relu(X, w, b):
    z = linear_predict(X, w, b)
    u = ReLU(z)
    return u

# 定義
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])
w = np.zeros(2)
b = 0.0
lr = 0.1

# 実行
'''
for epoch in range(100):
    z = linear_predict(X, w, b)
    y_pred = sigmoid(z)
    error = y_pred - y
    dw = (2 / len(y)) * (X.T @ error)
    db = (2 / len(y)) * sum(error)
    w = w - lr * dw
    b = b - lr * db
'''

# 二層までのニューラルネットワーク
def forward(X, w1, b1, w2, b2):
    z1 = linear_predict(X, w1, b1)
    u1 = ReLU(z1)
    z2 = linear_predict(u1, w2, b2)
    u2 = sigmoid(z2)
    return u2