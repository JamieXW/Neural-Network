import numpy as np

class Optimizer:
    def update(self, params: list[np.ndarray], grads: list[np.ndarray], learning_rate: float) -> None:
        """
        Updates parameters using gradients.

        Parameters:
            params (list of np.ndarray): List of parameters (e.g., weights, biases).
            grads (list of np.ndarray): List of gradients for each parameter.
            learning_rate (float): Learning rate for the update.
        """
        raise NotImplementedError("Update method not implemented")

class SGD(Optimizer):
    def update(self, params: list[np.ndarray], grads: list[np.ndarray], learning_rate: float) -> None:
        """
        Updates parameters using Stochastic Gradient Descent (SGD).

        Parameters:
            params (list of np.ndarray): List of parameters (e.g., weights, biases).
            grads (list of np.ndarray): List of gradients for each parameter.
            learning_rate (float): Learning rate for the update.
        """
        for param, grad in zip(params, grads):
            param -= learning_rate * grad