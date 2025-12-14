# Convolutional Neural Network (CNN) from Scratch

This project implements a Convolutional Neural Network (CNN) entirely from scratch using only `numpy` and `scipy`. It demonstrates the fundamental mechanics of deep learning layers—including Convolution, Dense (Fully Connected), Activation, and Reshaping—without relying on high-level frameworks like PyTorch or TensorFlow for the core logic.

## 📌 Features

* **Modular Layer Design:** Every component (Convolution, Dense, Activation) is a separate class inheriting from a base `Layer` class.
* **Convolutional Layer:** Implements 2D convolution with valid padding using `scipy.signal.correlate2d`.
* **Backpropagation:** Full implementation of forward and backward passes (gradients) for all layers.
* **Activation Functions:** Includes Tanh, Sigmoid, and Softmax implementations.
* **Loss Functions:** Implements Mean Squared Error (MSE) and Binary Cross Entropy.
* **MNIST Example:** Includes a driver script to train the network on a subset of the MNIST dataset (classifying digits 0 vs 1).

## 📂 Project Structure

* `layer`: Base class for all neural network layers.
* `convolutional`: Implementation of the Convolutional layer (forward/backward pass).
* `dense`: Implementation of the Fully Connected (Dense) layer.
* `activation`: Base class for activation functions.
* `activations`: Specific implementations of Tanh, Sigmoid, and Softmax.
* `reshape`: Layer to flatten/reshape tensors (e.g., between Conv and Dense layers).
* `losses`: Loss functions (MSE, Binary Cross Entropy) and their derivatives.
* `network`: Helper functions to train and predict using the network list.
* `main`: Driver script to load MNIST data and train the model.

## 🚀 Getting Started

### Prerequisites

You need Python installed along with the following libraries:

```bash
pip install numpy scipy keras tensorflow
