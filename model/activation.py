import numpy as np

class Activation:
    def forward(self, input):
        raise NotImplementedError("Forward method not implemented")

    def backward(self, output_gradient):
        raise NotImplementedError("Backward method not implemented")