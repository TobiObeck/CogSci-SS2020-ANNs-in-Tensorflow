import numpy as np

# Non-Linear Activation Functions


def sigmoid(x):
    """
    logistic function (is one type of sigmoid functions)
    basically a smoothed step function
    """
    return 1/(1+np.exp(-x))


def sigmoid_prime(x):
    """first derivative of sigmoid function"""
    return sigmoid(x)*(1-sigmoid(x))


def swish(x):
    return x * sigmoid(x)  # return x / (1+np.exp(-x))


def swish_prime(x):
    return swish(x) + sigmoid(x) * (1 - swish(x))


# The Dying ReLU problem - when inputs approach zero, or are negative,
# the gradient of the function becomes zero,
# the network cannot perform backpropagation and cannot learn.

def std_ReLU(x):
    """
    rectified linear units (ReLU)
    looks like --/
    0 for all x smaller than 0
    x for all x equal or greater than 0
    """
    return np.maximum(x, 0)


def std_ReLU_prime(x):
    if(x < 0):
        return 0
    else:
        return 1


def leaky_ReLU(x):
    """
    Prevents dying ReLU problem this variation of ReLU has a
    small positive slope in the negative area,
    so it does enable backpropagation, even for negative input values
    Otherwise like ReLU

    Disadvantages
    Results not consistent. leaky ReLU does not provide consistent predictions
    for negative input values.
    """
    return max(x * 0.1, x)


def leaky_ReLU_prime(x):
    if(x < 0):
        return 0.01
    else:
        return 1

###############################################################################
#  1. function that are not usable for backpropagation
#  the derivative of the function is a constant,
#  and has no relation to the input, X.
#  So it’s not possible to go back and understand
#  which weights in the input perceptrons can provide a better prediction.

# with linear activation functions,
# no matter how many layers in the neural network,
# the last layer will be a linear function of the first layer (because
# a linear combination of linear functions is still a linear function).
# So a linear activation function turns the neural network into just one layer.

# source :


"""
https://missinglink.ai/guides/
neural-network-concepts/7-types-neural-network-activation-functions-right/
"""


def binary_step(x):
    """Heavyside step function (denoted as theta or H)"""
    if(x < 0):
        return 0
    else:
        return 1


def linear(x):
    return x
