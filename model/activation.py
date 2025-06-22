import numpy as np

class Activation:
    def forward(self, input):
        raise NotImplementedError("Forward method not implemented")

    def backward(self, output_gradient):
        raise NotImplementedError("Backward method not implemented")
    
class ReLu(Activation):
    def forward(self, input):
        """
        applies the ReLu activation function to the input
        """
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient):
        """
        computes the gradient of the ReLu activation function
        with respect to the input during backpropagation.
        """
        return output_gradient * (self.input > 0)  # * 0 if input <= 0, 1 if input > 0