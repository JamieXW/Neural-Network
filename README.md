# Neural-Network

Building a Neural Network from Scratch

## Overview
This project implements a simple feedforward neural network from scratch in Python, using only NumPy. It is designed to classify images from the Fashion MNIST dataset.

## Features
- Custom implementation of Dense layers, activation functions (ReLU), and loss functions (Cross-Entropy, MSE)
- Stochastic Gradient Descent (SGD) optimizer
- Training loop with batch processing and accuracy evaluation
- One-hot encoding for multi-class classification

## Project Structure
```
Neural-Network/
├── main.py                # Entry point: trains and evaluates the model
├── model/
│   ├── layers.py          # Layer and Dense class definitions
│   ├── activation.py      # Activation functions (ReLU)
│   ├── loss.py            # Loss functions (CrossEntropy, MSE)
│   ├── optimizer.py       # Optimizer (SGD)
│   └── model.py           # Model class
├── FashionMNIST/
│   └── fashion-mnist_test.csv  # Fashion MNIST data (CSV)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Getting Started
1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
2. **Run the training script:**
   ```
   python main.py
   ```

## How It Works
- The network is built using the `Model` class, stacking `Dense` and `ReLU` layers.
- The Fashion MNIST data is loaded from CSV, normalized, and labels are one-hot encoded.
- The model is trained using mini-batch SGD and cross-entropy loss.
- After each epoch, the script prints the loss and accuracy.

## Customization
- You can change the network architecture in `main.py` by adding/removing layers.
- Adjust `epochs`, `batch_size`, and `learning_rate` for different training behavior.
- Add your own data or try different loss functions and optimizers.

## Requirements
- Python 3.8+
- numpy

## License
MIT License
