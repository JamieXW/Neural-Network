import numpy as np

class Loss:
    def forward(self, y_true, y_pred):
        raise NotImplementedError("Forward method not implemented")

    def backward(self, y_true, y_pred):
        raise NotImplementedError("Backward method not implemented")
    
class MeanSquaredError(Loss):
    def forward(self, y_true, y_pred):
        """
        Computes the Mean Squared Error loss.
        
        Parameters:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        
        Returns:
            float: The Mean Squared Error loss.
        """
        return np.mean(np.square(y_true - y_pred))

    def backward(self, y_true, y_pred):
        """
        Computes the gradient of the Mean Squared Error loss with respect to the predictions.
        
        Parameters:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        
        Returns:
            np.ndarray: Gradient of the loss with respect to the predictions.
        """
        return 2 * (y_pred - y_true) / y_true.size
    
class CrossEntropyLoss(Loss):
    def forward(self, y_true, y_pred):
        """
        Computes the Cross Entropy loss.
        
        Parameters:
            y_true (np.ndarray): True labels (one-hot encoded).
            y_pred (np.ndarray): Predicted probabilities.
        
        Returns:
            float: The Cross Entropy loss.
        """
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, y_true, y_pred):
        """
        Computes the gradient of the Cross Entropy loss with respect to the predictions.
        
        Parameters:
            y_true (np.ndarray): True labels (one-hot encoded).
            y_pred (np.ndarray): Predicted probabilities.
        
        Returns:
            np.ndarray: Gradient of the loss with respect to the predictions.
        """
        # Clip predictions to avoid division by zero
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        # For softmax + cross-entropy, the gradient is (y_pred - y_true) / batch_size
        return (y_pred - y_true) / y_true.shape[0]