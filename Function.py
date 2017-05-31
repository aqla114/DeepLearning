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

def MeanSquaredError(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def CrossEntropyError(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y) / batch_size)

def NumericalDiff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / 2*h

def NumericalGradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 値を元に戻す
        it.iternext()

    return grad
