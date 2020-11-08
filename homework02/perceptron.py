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
        self.alpha = 0.01  # learning rate how fast? slow: 0.01 faster: 1
        self.weights = np.random.randn(input_unit_length)
        self.bias = np.random.randn(1)  # we only need one bias
        # np.random.random()
        # np.randon.randint(3)

    # Calculates the drive / activation.
    # will the neuron be activated and pass 1 or 0 forward?
    # You could use @ as a matrix multiplication command.
    # weighted_sum = self.weights @ data + self.bias
    def forward_step(self, input_data):
        weighted_sum = np.dot(self.weights, input_data) + self.bias
        return self.threshold(weighted_sum)

    # 2. Returns a 1 or a 0, depending on whether the perceptron 
    # surpassed the threshold. int(...) is casting a bool to int
    def threshold(self, weighted_sum):
        return int(weighted_sum >= 0)

    # 1. forward step
    # 2. calculate weight updates
    # 3. bias updates
    # times 1, because bias is unweighted
    def training_step(self, input, label):
        prediction = self.forward_step(input)
        self.weights += self.alpha * \
            (label - prediction) * input  # plus or minus equals??
        self.bias += self.alpha * (label - prediction) * 1
        # + error term needed here somewhere?


possible_inputs = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])

# targets
t_and = np.array([0, 0, 0, 1])
t_or = np.array([0, 1, 1, 1])
t_and = np.array([1, 1, 1, 0])
t_or = np.array([1, 0, 0, 0])
t_xor = np.array([0, 1, 1, 0])

# training
# target that we are training on
target = t_and

# new perception instance
INPUT_LENGTH = np.size(possible_inputs, 1) # length of sub-array (2)
perceptron = Perceptron(INPUT_LENGTH)

# training loop
steps = []
accuracies = []

for i in range(500):
    steps.append(i)

    # draw a random sample from data
    index = np.random.randint(len(possible_inputs))
    sample = possible_inputs[index]
    label = target[index]

    # perform a trianing step
    perceptron.training_step(sample, label)

    # how does our  perceptroin perform?
    # perfonmrabnce overall four possible inputs

    accuracy_sum = 0
    for j in range(len(possible_inputs)):
        output = perceptron.forward_step(possible_inputs[j])
        accuracy_sum += int(output == target[j])
    accuracy = accuracy_sum/4
    accuracies.append(accuracy)

for n in range(len(possible_inputs)):
    output = perceptron.forward_step(possible_inputs[n])
    print("Input: %s \t True Label: %d \t Perceptron's Prediction: %d" %
          (np.array2string(possible_inputs[n]), target[n], output))

# visualize training
plt.figure()
plt.plot(steps, accuracies)
plt.xlabel("Training Steps")
plt.ylabel("Accuracy")
#plt.ylim([-0.1, 1.2])
plt.show()
