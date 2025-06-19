import numpy as np

class Loss:
    def forward(self, y_true, y_pred):
        raise NotImplementedError("Forward method not implemented")

    def backward(self, y_true, y_pred):
        raise NotImplementedError("Backward method not implemented")