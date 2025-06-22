import numpy as np

class layer:
    def forward(self, input):
        """
        Computes the output of the layer for a given input.
        Should be implemented by subclasses.

        Parameters:
            input (np.ndarray): Input data to the layer.

        Returns:
            np.ndarray: Output of the layer.
        """
        raise NotImplementedError("Forward method not implemented")
    
    def backward(self, output_gradient):
        """
        Computes the gradient of the loss with respect to the input of the layer.
        Should be implemented by subclasses.

        Parameters:
            output_gradient (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        raise NotImplementedError("Backward method not implemented")
    
    def update_params(self, learning_rate):
        """
        Updates the parameters of the layer, if any.
        Should be implemented by subclasses if the layer has parameters.

        Parameters:
            learning_rate (float): The learning rate for parameter updates.
        """
        pass

class Dense(layer):
    def __init__(self, input_size, output_size):
        """
        A layer in the neural network 

        Parameters:
            weights (np.ndarray): Weights of the layer, initialized randomly.
            biases (np.ndarray): Biases of the layer, initialized to zeros.
        """
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.input = None
        self.grad_weights = None
        self.grad_biases = None

    def forward(self, input):
        """
        Performs the forward pass for the Dense layer.

        Parameters:
            input (np.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            np.ndarray: Output data of shape (batch_size, output_size).
        """
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, output_gradient):
        """
        Performs the backward pass, computing gradients for weights, biases, and input.

        Parameters:
            output_gradient (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        self.grad_weights = np.dot(self.input.T, output_gradient)
        self.grad_biases = np.sum(output_gradient, axis=0, keepdims=True)
        return np.dot(output_gradient, self.weights.T)

    def update_params(self, learning_rate):
        """
        Updates the weights and biases using the computed gradients.

        Parameters:
            learning_rate (float): The learning rate for parameter updates.
        """
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases