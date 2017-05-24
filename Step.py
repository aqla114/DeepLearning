import numpy as np

def Step(x):
    y = x > 0
    return y.astype(np.int)
