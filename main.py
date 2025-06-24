import numpy as np 
from model.layers import Dense
from model.model import Model
from model.activation import ReLu
from model.activation import Softmax
from model.optimizer import SGD
from model.loss import CrossEntropyLoss

def load_fashion_mnist_csv(path):
    # Loads fashion MNIST from csv, returns (x, y)
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    x = data[:, 1:] / 255.0 
    y = data[:, 0].astype(int) 
    return x, y

def one_hot(y, num_classes):
    # Converts labels to one-hot encoding
    return np.eye(num_classes)[y]

def main():
    x_train, y_train = load_fashion_mnist_csv('./FashionMNIST/fashion-mnist_test.csv')
    num_classes = 10
    y_train_one_hot = one_hot(y_train, num_classes)

    model = Model()
    model.add(Dense(784, 256))
    model.add(ReLu())
    model.add(Dense(256, 128))
    model.add(ReLu())
    model.add(Dense(128, 64))
    model.add(ReLu())
    model.add(Dense(64, num_classes))
    model.add(Softmax())


    loss_function = CrossEntropyLoss()

    epochs = 5 # term for number of passes through the training datatset
    batch_size = 64 # the number of training samples used in one iteration of training
    learning_rate = 0.001 # the step size of the optimizer when updating the model parameters

    for epoch in range(epochs):
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train_one_hot = y_train_one_hot[indices]

        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train_one_hot[i:i + batch_size]

            # Forward pass
            predictions = model.forward(x_batch)
            loss = loss_function.forward(y_batch, predictions)

            # # debug prints
            # print("x_batch shape:", x_batch.shape)
            # print("y_batch shape:", y_batch.shape)
            # print("predictions min/max:", predictions.min(), predictions.max())
            # print("loss:", loss)
            # if np.isnan(predictions).any() or np.isnan(loss):
            #     print("NaN detected!")
            #     break

            # Backward pass
            gradient = loss_function.backward(y_batch, predictions)
            model.backward(gradient)

            # Update parameters
            model.update_params(learning_rate)
        
        # Evaluate accuracy after each epoch
        all_preds = model.forward(x_train)
        pred_labels = np.argmax(all_preds, axis=1)
        true_labels = np.argmax(y_train_one_hot, axis=1)
        accuracy = np.mean(pred_labels == true_labels)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%')
    
if __name__ == "__main__":
    main()