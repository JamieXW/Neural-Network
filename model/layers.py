import numpy as np

class layer:
    def forward(self, input):
        raise NotImplementedError("Forward method not implemented")
    
    def backward(self, output_gradient):
        raise NotImplementedError("Backward method not implemented")
    
    def update_params(self, learning_rate):
        pass
    