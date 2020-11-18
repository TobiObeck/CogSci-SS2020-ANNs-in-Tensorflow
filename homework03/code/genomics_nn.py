# https://www.tensorflow.org/datasets/catalog/genomics_ood

# input = genomic sequence consisting of characters A, C, G, T (nucleotides)
# each sequence = 250 characters long. so 256 ACGTGTCAGCTAGCT... characters
# The sample size of each class is
#   - 100,000 in the training
#   - 10,000 for the validation and test sets.

"""
For each example, the features include:
- seq: the input DNA sequence composed by {A, C, G, T}.
- label: the name of the bacteria class.
- seq_info: the source of the DNA sequence, i.e., the genome name,
  NCBI accession number, and the position where it was sampled from.
- domain: if the bacteria is in-distribution (in), or OOD (ood)

FeaturesDict({
    'domain': Text(shape=(), dtype=tf.string),
    'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=130),
    'seq': Text(shape=(), dtype=tf.string),
    'seq_info': Text(shape=(), dtype=tf.string),
})
"""
# https://github.com/google-research/google-research/tree/master/genomics_ood

# Task: classify a sequence as 1 of 10 bacteria population
# (in-distribution classes, discovered before the year 2011)

# Note that training, validation, and test data are provided
# for the in-distribution classes

# Further bacteria populations that are not relevant for this task,
# because they are out-of-distribution (OOD):
#   60 bacteria classes (2011-2016) as  for validation
#   60 different bacteria classes discovered after 2016 as OOD for test
#   in total 130 bacteria classes

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# function definitions
######################

def print_sample_for_first_n_records(tf_data_dataset, record_size):
    """
    peek at contents of the first n data records
    """

    print(type(tf_data_dataset))
    print(tf.data.experimental.cardinality(tf_data_dataset))

    for i in range(record_size):
        temp_record = tf_data_dataset.as_numpy_iterator().next()
        print("domain:", temp_record['domain'])
        print("label:", temp_record['label'])
        print("seq:", temp_record['seq'])
        print("seq_info:", temp_record['seq_info'], "\n")
    print("\n")


def extract_shuffled_batch(tf_data_dataset, batch_size):
    temp = tf_data_dataset.batch(batch_size)
    return temp.shuffle(buffer_size=batch_size)


def onehotify(tensor):
    vocab = {'A': '1', 'C': '2', 'G': '3', 'T': '0'}
    for key in vocab.keys():
        tensor = tf.strings.regex_replace(tensor, key, vocab[key])
    split = tf.strings.bytes_split(tensor)
    labels = tf.cast(tf.strings.to_number(split), tf.uint8)
    onehot = tf.one_hot(labels, 4)
    onehot = tf.reshape(onehot, (-1,))
    return onehot

# main instructions
######################

# tf.data.Dataset https://www.tensorflow.org/api_docs/python/tf/data/Dataset
# input pipelines: https://www.tensorflow.org/guide/data
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map

### LOADING SEPARATELY AS KIND OF DICTIONARY, NOT TUPLE

train_raw_all = tfds.load('genomics_ood', split='train')
test_raw_all = tfds.load('genomics_ood', split='test',)

train_batch_size = 100_000
test_batch_size = 1000

train_raw_all = train_raw_all.prefetch(train_batch_size)
test_raw_all = test_raw_all.prefetch(test_batch_size)

# print(train_raw_all)
# print_sample_for_first_n_records(train_raw_all, record_size=2)
# print_sample_for_first_n_records(test_raw_all, record_size=2)


train_encoded_inputs = train_raw_all.map(lambda t : onehotify(t['seq']))
train_encoded_labels = train_raw_all.map(lambda t : tf.one_hot(t['label'], 10))
print(train_encoded_inputs, train_encoded_labels)

test_encoded_inputs = test_raw_all.map(lambda t : onehotify(t['seq']))
test_encoded_labels = test_raw_all.map(lambda t : tf.one_hot(t['label'], 10))
print(test_encoded_inputs, test_encoded_labels)

# input_gen = (x for x in train_encoded_inputs)
# label_gen = (x for x in train_encoded_labels)
# for i, some_stuff in enumerate(label_gen): # next(generator_input)
#     print(i, some_stuff)
#     if(i == 3): break

train_prepared = tf.data.Dataset.zip((train_encoded_inputs, train_encoded_labels))
train_batched = train_prepared.batch(train_batch_size)
train_shuffled_batch = train_batched.shuffle(buffer_size=train_batch_size)

test_prepared = tf.data.Dataset.zip((test_encoded_inputs, test_encoded_labels))
test_batched = test_prepared.batch(test_batch_size)
test_shuffled_batch = test_batched.shuffle(buffer_size=test_batch_size)


from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

class Model(Model):

    def __init__(self):
        super(Model, self).__init__()
        # Define the three layers.
        self.hidden_layer_1 = tf.keras.layers.Dense(units=256,
                                               activation=tf.keras.activations.sigmoid
                                               )
        self.hidden_layer_2 = tf.keras.layers.Dense(units=256,
                                               activation=tf.keras.activations.sigmoid
                                               )
        self.output_layer = tf.keras.layers.Dense(units=10,
                                               activation=tf.keras.activations.softmax
                                               )
    def call(self, x):
        # Define the forward step.
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)
        return x

def train_step(model, input, target, loss_function, optimizer):
  # loss_object and optimizer_object are instances of respective tensorflow classes
  with tf.GradientTape() as tape:
    prediction = model(input)
    loss = loss_function(target, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

def test(model, test_data, loss_function):
  # test over complete test data

  test_accuracy_aggregator = []
  test_loss_aggregator = []

  for (input, target) in test_data:
    prediction = model(input)
    sample_test_loss = loss_function(target, prediction)
    sample_test_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
    sample_test_accuracy = np.mean(sample_test_accuracy)
    test_loss_aggregator.append(sample_test_loss.numpy())
    test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

  test_loss = np.mean(test_loss_aggregator)
  test_accuracy = np.mean(test_accuracy_aggregator)

  return test_loss, test_accuracy


tf.keras.backend.clear_session()

### Hyperparameters
num_epochs = 10
learning_rate = 0.1
running_average_factor = 0.95

# Initialize the model.
model = Model()
# Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
# Initialize the optimizer: Adam with default parameters. Check out 'tf.keras.optimizers'
optimizer = tf.keras.optimizers.SGD(learning_rate)

# Initialize lists for later visualization.
train_losses = []

test_losses = []
test_accuracies = []

#testing once before we begin
test_loss, test_accuracy = test(model, test_shuffled_batch, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

#check how model performs on train data once before we begin
train_loss, _ = test(model, train_shuffled_batch, cross_entropy_loss)
train_losses.append(train_loss)

# We train for num_epochs epochs.
for epoch in range(num_epochs):
    print('Epoch: __ ' + str(epoch))

    train_shuffled_batch = train_shuffled_batch.shuffle(buffer_size=128)
    test_shuffled_batch = test_shuffled_batch.shuffle(buffer_size=128)

    #training (and checking in with training)
    running_average = 0
    for (input,target) in train_shuffled_batch:
        train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
        running_average = running_average_factor * running_average  + (1 - running_average_factor) * train_loss
    train_losses.append(running_average)

    #testing
    test_loss, test_accuracy = test(model, test_shuffled_batch, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)


# Visualize accuracy and loss for training and test data.
# One plot training and test loss.
# One plot training and test accuracy.
plt.figure()
line1, = plt.plot(train_losses)
line2, = plt.plot(test_losses)
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.legend((line1,line2),("training","test"))
plt.show()

plt.figure()
line1, = plt.plot(test_accuracies)
plt.xlabel("Training steps")
plt.ylabel("Accuracy")
plt.show()