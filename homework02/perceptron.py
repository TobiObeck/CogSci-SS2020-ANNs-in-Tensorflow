# implementing a perceptron
# train it on logical gates

# and -> only active if both inputs are true
# otherwise it shouldn't

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:

    """
    initializing a perceptron with a normal distributed bias
    and a single bias
    """

    def __init__(self, input_unit_length):
        # how many input values are going into the perceptron
        self.input_unit_length = input_unit_length
        self.alpha = 0.1  # learning rate how fast? slow: 0.01 faster: 1
        self.weights = np.random.randn(input_unit_length)
        self.bias = np.random.randn(1)  # we only need one bias
        # np.random.random()
        # np.randon.randint(3)
        self.inputs = 0
        self.drive = 0

    # Calculates the drive / activation.
    # activation = every input is multiplied with every weight + bias
    # wrapped by activation function
    # will the neuron be activated and pass 1 or 0 forward?
    # You could use @ as a matrix multiplication command.
    # weighted_sum or drive = self.weights @ data + self.bias
    def forward_step(self, input_data):
        self.drive = np.dot(self.weights, input_data) + self.bias # drive
        self.inputs = input_data
        return self.threshold(self.drive)

    # 2. Returns a 1 or a 0, depending on whether the perceptron
    # surpassed the threshold. int(...) is casting a bool to int
    #def threshold(self, drive):
    #    return int(drive >= 0)

    def threshold(self, weighted_sum):        
        return sigmoid(weighted_sum)
    
    # this method was given in live session
    # 1. forward step
    # 2. calculate weight updates
    # 3. bias updates
    # times 1, because bias is unweighted
    def training_step(self, sample, label):
        "Premature implementation"
        prediction = self.forward_step(sample)
        self.weights += self.alpha * (label - prediction) * sample
        self.bias += self.alpha * (label - prediction) * 1
        # + error term needed here somewhere?

    def update(self, delta):        
        gradient_weights = delta * self.inputs
        gradient_bias = delta
        self.weights -=  self.alpha * gradient_weights
        self.bias -= self.alpha * gradient_bias

    def get_prediction(self, input_data):
        drive = np.dot(self.weights, input_data) + self.bias # drive
        return self.threshold(drive)

    def debug_get_weights(self):
        return self.weights

    def debug_get_bias(self):
        return self.bias


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

"""
possible_inputs = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])

# possible targets
t_and = np.array([0,0,0,1])
t_or = np.array([0,1,1,1])
t_nand = np.array([1,1,1,0])
t_nor = np.array([1,0,0,0])
t_xor = np.array([0,1,1,0])

# training
# target that we are training on
target = t_and # input x_i

# new perception instance
INPUT_LENGTH = np.size(possible_inputs, 1)  # length of sub-array (2)
perceptron = Perceptron(INPUT_LENGTH)

# training loop
steps = []
accuracies = []

for i in range(1500):
    steps.append(i)

    # draw a random sample from data
    rand_index = np.random.randint(len(possible_inputs))

    sample = possible_inputs[rand_index] # t_i
    label = target[rand_index] # x_i

    # perform a trianing step
    perceptron.training_step(sample, label)

    # how does our  perceptroin perform?
    # performance overall four possible inputs

    accuracy_sum = 0
    for k in range(len(possible_inputs)):
        output = perceptron.forward_step(possible_inputs[k])
        accuracy_sum += int(output == target[k])
    accuracy = accuracy_sum/4
    accuracies.append(accuracy)

for n in range(len(possible_inputs)):
    output = perceptron.forward_step(possible_inputs[n])    
    print("Input: %s \t True Label: %d \t Perceptron's Prediction: %d" %
          (np.array2string(possible_inputs[n]), target[n], output))

print("weights", perceptron.debug_get_weights())
print("bias", perceptron.debug_get_bias())

# visualize training
plt.figure()
plt.plot(steps, accuracies)
plt.xlabel("Training Steps")
plt.ylabel("Accuracy")
plt.ylim([-0.1, 1.2])
plt.show()
"""