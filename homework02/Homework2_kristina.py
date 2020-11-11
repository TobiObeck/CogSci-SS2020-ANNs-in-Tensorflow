import numpy as np
import matplotlib.pyplot as plt

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# targets
t_and = np.array([0, 0, 0, 1])
t_or = np.array([0, 1, 1, 1])
t_nand = np.array([1, 1, 1, 0])
t_nor = np.array([1, 0, 0, 0])
t_xor = np.array([0, 1, 1, 0])


# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidprime(x):  # this is the first derivate of the sigmoid function
    return sigmoid(x) * (1 - sigmoid(x))


class Perceptron:
    """insert documentation here"""

    def __init__(self, input_units):
        self.input_units = input_units
        self.alpha = 0.01
        self.weights = np.random.randn(input_units)
        self.bias = np.random.randn(1)
        self.alpha = 0.01
        # these will be needed for the values from the neuron later, so setting up the variables to store them
        self.inputs = 0
        self.drive = 0

    def forward_step(self, inputs):
        self.inputs = inputs
        self.drive = self.weights @ inputs + self.bias  # calculate the drive
        return sigmoid(self.drive)  # return the activation function with drive as its argument

    def update(self, delta):  # delta is the error term, it will be obtained from back-propagation
        weights_gradient = delta * self.inputs
        bias_gradient = delta
        self.weights -= self.alpha * weights_gradient
        self.bias -= self.alpha * bias_gradient


class MLP:

    def __init__(self):
        # we need to initialise the perceptrons for the hidden layer of our MLP..
        self.hidden_layer = [
            Perceptron(input_units=2),
            Perceptron(input_units=2),
            Perceptron(input_units=2),
            Perceptron(input_units=2)
        ]
        # ..and also one output neuron
        self.output_neuron = Perceptron(input_units=4)
        # output variable will later store our output
        self.output = 0

    def forward_step(self, inputs):
        # compute forward step = activation for each neuron in the hidden layer...
        hidden_layer_activations = np.array([p.forward_step(inputs) for p in self.hidden_layer])
        hidden_layer_activations = np.reshape(hidden_layer_activations, newshape=(-1))
        # ...and compute the activation of the output neuron
        self.output = self.output_neuron.forward_step(hidden_layer_activations)

    def backprop_step(self, inputs, target):
        # first, compute the delta at the output neuron according to the formula
        output_delta = - (target - self.output) * sigmoidprime(self.output_neuron.drive)
        # now update the parameters of  the output neuron by inserting the obtained delta
        self.output_neuron.update(output_delta)
        # next, compute the deltas for the hidden neurons
        hidden_deltas = [output_delta * sigmoidprime(p.drive) * self.output_neuron.weights[i] for i, p in
                         enumerate(self.hidden_layer)]
        # again, update the parameters for all neurons in the hidden layer
        for i, p in enumerate(self.hidden_layer):
            p.update(hidden_deltas[i])


# TRAINING PART
mlp = MLP()
# initialize lists to store epochs, loss, accuracy of the predictions
steps = [] # a.k.a epochs
losses = []
accuracies = []

for i in range(1000):
    steps.append(i)

    # 1. Draw a random sample from x and the corresponding t. Check 'np.random.randint'.
    # index = np.random.randint(len(x))
    # sample = x[index]
    # label = t_xor[index]

    accuracy = 0
    loss = 0

    for k in range(len(x)):
        sample = x[k]
        label = t_xor[k]
        mlp.forward_step(sample)
        mlp.backprop_step(sample, label)

        accuracy += int(float(mlp.output >= 0.5) == label)
        loss += (label - mlp.output) ** 2  # mean squared error

    accuracies.append(accuracy/4)
    losses.append(loss)

plt.figure()
plt.plot(steps, losses)
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(steps, accuracies)
plt.xlabel("Training Steps")
plt.ylabel("Accuracy")
plt.ylim([-0.1, 1.2])
plt.show()
