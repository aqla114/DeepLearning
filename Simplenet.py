import sys, os
sys.path.append(os.pardir)
import numpy as np
from Function import Softmax, CrossEntropyError
from Function import NumericalGradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) #ガウス分布で初期化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = Softmax(z)
        loss = CrossEntropyError(y, t)

        return loss