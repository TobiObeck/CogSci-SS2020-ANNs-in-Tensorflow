import numpy as np


class Perceptron:
    # Initialize random weights and a random bias term.
    # The weights with mean 0 and stddev 0.5.
    # The bias with mean 0 and stddev 0.05. Check 'np.random.normal()'.
    def __init__(self, learning_rate, input_length, activation_function):
        self.inputs = 0
        self.drive = 0
        self.alpha = learning_rate

        self.weights = np.random.normal(loc=0, scale=0.5, size=input_length)
        self.bias = np.random.normal(loc=0, scale=0.05, size=1)

        self.activation_function = activation_function

    def forward_step(self, inputs):
        self.inputs = inputs
        # Calculate the drive and store it in the corresponding variable.
        self.drive = self.weights @ inputs + self.bias
        # Return the activation.
        return self.activation_function(self.drive)

    def update(self, delta):
        # We will call this function to update the parameters
        # for this specific perceptron.
        # The function is provide with a delta.
        # So you only need to compute the gradients perform the update.

        # Compute the gradients for weights and bias.
        gradient_weights = delta * self.inputs
        gradient_bias = delta
        # Update weights and bias.
        self.weights -= self.alpha * gradient_weights
        self.bias -= self.alpha * gradient_bias
