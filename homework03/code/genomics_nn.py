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

# alternative way of loading as one liner
# train_raw, test_raw = tfds.load('genomics_ood', split=['train', 'test'])

# tf.data.Dataset https://www.tensorflow.org/api_docs/python/tf/data/Dataset
train_tuple = tfds.load('genomics_ood', split='train', as_supervised=True)
"""
train_raw_all = tfds.load('genomics_ood', split='train')
test_raw_all = tfds.load('genomics_ood', split='test',)

print(train_raw_all)
print_sample_for_first_n_records(train_raw_all, record_size=2)
print_sample_for_first_n_records(test_raw_all, record_size=2)
"""

"""
generator = (x for x in train_tuple)

for i, raw_tuple in enumerate(generator): # next(generator_input)
    input_i, target_i = raw_tuple
    # print(input_i, "\n")
    # print(target_i, "\n\n")
    if(i == 3):
        break
"""

# train_slice = tf.data.Dataset.from_tensor_slices(train_tuple)
# print(train_slice)
tensors_tuple = tf.data.Dataset.from_tensors(train_tuple)
print("SLICE")
print(tensors_tuple, "\n")

print("shape:", tensors_tuple)

# mapped_stuff = tensors_tuple(0).map(lambda t : onehotify(t))
# print(mapped_stuff)

for i, tensor in enumerate(train_raw_all):
    # print(tensor) # print(tensor[['seq']])
    # print(tensor['seq'], "\n")
    # print(tensor['label'], "\n\n")
    if(i == 5):
        break


"""
# train_tensor = tf.data.Dataset.from_tensors(train_raw_all)
train_hotified = onehotify(train_tensor)
#print_sample_for_first_n_records(train_hotified, record_size=2)

train_batch = extract_shuffled_batch(train_raw_all, batch_size=4)  # 100_000)
test_batch = extract_shuffled_batch(train_raw_all, batch_size=10)  # 1000)

print_sample_for_first_n_records(train_batch, record_size=2)

"""

# bytes_string.decode('UTF-8')