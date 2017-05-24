import numpy as np

def Step(x):
    y = x > 0
    return y.astype(np.int)

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def Identity(x):
    return x

def Softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c) #オーバーフロー対策
    sum_exp_x = np.sum(exp_x)

    return exp_x / sum_exp_x
