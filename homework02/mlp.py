from perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt

# multi layered perceptron
class MLP(object):

  layers = []

  def __init__(self, input_length, network_shape):
    for layer_length in network_shape:
      perceptrons = []
      for _ in range(layer_length):
        perceptron = Perceptron(input_length)
        perceptrons.append(perceptron)        
      self.layers.append(perceptrons)
    print(self.layers)
    self.output = 0

  #def train(self, x, t):
  #  """
  #  x=inputs (data sample)
  #  t=target of the inputs
  #  """
  #  self.forward_step(x)
  #  # self.backprop_step(x,t)

  def forward_step(self, x):
    # print(x, "forward_step not implemented")
    activations = []
    for i in range(len(self.layers[len(self.layers)-1])):
      activations.append(self.layers[len(self.layers)-1][i].forward_step(x))
    activations = np.array(activations)    
    self.output = activations
    return activations

  def backprop_step(self, x,t):
    print(x, t, "backprop_step not implemented")    

  def get_prediction(self, x):    
    prediction_sum = 0
    for i in range(len(self.layers[len(self.layers)-1])):
      prediction_sum += self.layers[len(self.layers)-1][i].get_prediction(x)
    prediction_sum = prediction_sum / len(self.layers[len(self.layers)-1])
    self.output = prediction_sum 
    return prediction_sum
    #return non mutating forward step

  def DEBUG_network(self):
    print("output", self.output)
    print("layer length", len(self.layers))
    for i in range(len(self.layers[len(self.layers)-1])):
      print(self.layers[len(self.layers)-1][i])
      

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
target = t_and

my_network = MLP(2, [2,4,1])

steps = []
epochs = []
losses = []
accuracies = []
accuracies_per_epoch = []

for epoch in range(500):  
  epochs.append(epoch)
  steps.append(epoch)

  accuracy_buffer = 0
  loss_buffer = 0

  for index in range(len(possible_inputs)):
    
    #rand_index = np.random.randint(len(possible_inputs))

    input_i = possible_inputs[index]
    target_i = target[index]  

    my_network.forward_step(input_i) #my_network.train(input_i, target_i)

    accuracy_buffer += int(float(my_network.output>=0.5) == target_i)
    loss_buffer += (target_i-my_network.output)**2        
  accuracies.append(accuracy_buffer/4.0)
  losses.append(loss_buffer)

  accuracy_sum = 0
  for k in range(len(possible_inputs)):
    output = my_network.forward_step(possible_inputs[k])
    if(output == target[k]):
      accuracy_sum += 1    
  accuracies_per_epoch.append(accuracy_sum/4)
  
for n in range(len(possible_inputs)):
    output = my_network.forward_step(possible_inputs[n])
    print("Input: %s \t True Label: %d \t Perceptron's Prediction: %d" %
          (np.array2string(possible_inputs[n]), target[n], output))

"""
accuracy_sum = 0


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
"""

# print(steps, accuracies)

# visualize training
plt.figure()
#plt.plot(steps, accuracies)
plt.plot(epochs, accuracies_per_epoch)
plt.xlabel("Training Steps")
plt.ylabel("Accuracy")
#plt.ylim([-0.1, 1.2])
plt.show()

my_network.DEBUG_network()