from mlp import MLP
import numpy as np
import matplotlib.pyplot as plt
from activation_functions import (
    sigmoid, sigmoid_prime,
    std_ReLU, std_ReLU_prime,
    leaky_ReLU, leaky_ReLU_prime,
    swish, swish_prime
)

# possible targets
t_and = np.array([0, 0, 0, 1], dtype=np.float32)
t_or = np.array([0, 1, 1, 1], dtype=np.float32)
t_nand = np.array([1, 1, 1, 0], dtype=np.float32)
t_nor = np.array([1, 0, 0, 0], dtype=np.float32)
t_xor = np.array([0, 1, 1, 0], dtype=np.float32)

input_pairs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
true_target = t_xor

sigmoid_funcs = (sigmoid, sigmoid_prime)  # LR 1 and 3000 epochs
swish_funcs = (swish, swish_prime)  # can work with slow LR of 0.1-0.03 and 3000 epochs
relu_funcs = (std_ReLU, std_ReLU_prime)  # do not work that well
leaky_relu_funcs = (leaky_ReLU, leaky_ReLU_prime)  # do not work that well

learning_rate = 1
mlp = MLP(learning_rate, sigmoid_funcs)
epochs_length = 1000
# learning_rate = 0.03
# mlp = MLP(learning_rate, swish_funcs)
# epochs_length = 10000

epochs = []
losses = []
accuracies_per_epoch = []


# is_accuracy_good_enough = False

for epoch in range(epochs_length):  # 500-5000
    epochs.append(epoch)

    accuracy_accumulator = 0
    loss_buffer = 0

    # Training loop.
    for i in range(4):
        x = input_pairs[i]
        t = true_target[i]

        mlp.train(x, t)

        accuracy_accumulator += int(float(mlp.output >= 0.5) == t)
        loss_buffer += (t-mlp.output)**2

    accuracies_per_epoch.append(accuracy_accumulator/4.0)
    losses.append(loss_buffer)

    if(epoch % (epochs_length / 10) == 0):
        total_loss = 0
        for n in range(len(input_pairs)):
            x = input_pairs[n]
            t = true_target[n]
            mlp.train(x, t)
            diff = t.astype(np.float32) - mlp.output
            total_loss += abs(diff)
            print(f"Input: {np.array2string(x)}",
                  f"True Label: {t}",
                  f"MLP's Prediction: {round(float(mlp.output), 3)}",
                  f"Diff: {round(float(diff), 3)}")
        average_loss = total_loss/4
        print(f"In epoch: {epoch}: ",
              f"avg loss of: {round(float(average_loss), 3)}\n")
        if(average_loss < 0.03):
            break  # is_accuracy_good_enough = True

# Visualize the training progress. Loss and accuracy.
# If the performance does not reach 100% just rerun the cell above.
plt.figure()
plt.plot(epochs, losses)
plt.xlabel("epoch no.")
plt.ylabel("Loss per epoch")
plt.show()

plt.figure()
plt.plot(epochs, accuracies_per_epoch)
plt.xlabel("epoch no.")
plt.ylim([-0.1, 1.2])
# plt.ylim([-0.1, 1.2])
plt.ylabel("Accuracy per epoch")
plt.show()
