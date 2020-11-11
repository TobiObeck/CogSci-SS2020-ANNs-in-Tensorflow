import numpy as np
from perceptron import Perceptron
from activation_functions import sigmoid, sigmoid_prime


class MLP:
    def __init__(
        self,
        learning_rate=1,
        activation_funcs=(sigmoid, sigmoid_prime)
    ):
        self.activation_func = activation_funcs[0]
        self.activation_func_prime = activation_funcs[1]
        self.output = 0

        self.initialize_hidden_layer(learning_rate)
        self.initialize_output_perceptron(learning_rate)

    def initialize_hidden_layer(self, learning_rate):
        hidden_layer_input_length = 2
        layer_length = 4
        self.hidden_layer = [
            Perceptron(learning_rate, hidden_layer_input_length, sigmoid)
            for _ in range(layer_length)
        ]

    def initialize_output_perceptron(self, learning_rate):
        output_layer_input_length = 4
        self.output_perceptron = Perceptron(
            learning_rate,
            output_layer_input_length,
            self.activation_func
        )

    def train(self, inputs, target):
        """performs forward and backward step with one labeled data sample"""
        self.forward_step(inputs)
        self.backprop_step(inputs, target)

    # Compute the activations for the hidden layer.
    # You might need to reshape ((4,1)->(4,)) the resulting array
    #  to feed it to the output perceptron.
    #  Check 'np.reshape(arr, newshape=(-1)).'
    # Compute the activation of the output perceptron
    #  and store it in 'self.output'.
    def forward_step(self, inputs):
        activations_hidden = np.array(
            [p.forward_step(inputs) for p in self.hidden_layer]
        )
        activations_hidden = np.reshape(activations_hidden, newshape=(-1))
        self.output = self.output_perceptron.forward_step(activations_hidden)

    # Use the Sum-squared error (lecture 3) as the loss function.
    #  Compute the delta at the output perceptron.
    # Update the parameters of  the output perceptron.
    # Compute the deltas for the hidden perceptrons.
    # Update the parameters for all four perceptrons in the hidden layer.
    def backprop_step(self, inputs, target):

        output_delta = -(target - self.output) * \
            self.activation_func_prime(self.output_perceptron.drive)

        self.output_perceptron.update(output_delta)

        hidden_deltas = [
            output_delta
            * self.activation_func_prime(p.drive)
            * self.output_perceptron.weights[i]
            for i, p in enumerate(self.hidden_layer)
        ]
        for i, p in enumerate(self.hidden_layer):
            p.update(hidden_deltas[i])
