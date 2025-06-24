import numpy as np

class Activation:
    def forward(self, input):
        raise NotImplementedError("Forward method not implemented")

    def backward(self, output_gradient):
        raise NotImplementedError("Backward method not implemented")
    
    def update_params(self, learning_rate):
        # Activation layers have no parameters to update
        pass
    

class Softmax(Activation):
    def __init__(self):
        self.output = None

    def forward(self, input):
        shifted = input - np.max(input, axis=1, keepdims=True)
        exps = np.exp(shifted)
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output
    
    def backward(self, output_gradient):
        return output_gradient  

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