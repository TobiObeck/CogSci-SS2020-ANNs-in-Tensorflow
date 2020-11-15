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

### LOADING SEPARATELY AS KIND OF DICTIONARY, NOT TUPLE

train_raw_all = tfds.load('genomics_ood', split='train')
test_raw_all = tfds.load('genomics_ood', split='test',)
train_batch_size = 20 # 100_000
test_batch_size = 20 # 1000
train_raw_all = train_raw_all.prefetch(train_batch_size)
test_raw_all = test_raw_all.prefetch(train_batch_size)


# print(train_raw_all)
# print_sample_for_first_n_records(train_raw_all, record_size=2)
# print_sample_for_first_n_records(test_raw_all, record_size=2)

encoded_inputs = train_raw_all.map(lambda t : onehotify(t['seq']))
encoded_labels = train_raw_all.map(lambda t : tf.one_hot(t['label'], 10))
print(encoded_inputs, encoded_labels)

input_gen = (x for x in encoded_inputs)
label_gen = (x for x in encoded_labels)

for i, some_stuff in enumerate(label_gen): # next(generator_input)
    print(i, some_stuff)
    if(i == 3): break

train_prepared = tf.data.Dataset.zip((encoded_inputs, encoded_labels))
print(train_prepared)

train_batched = train_prepared.batch(train_batch_size)
train_shuffled_batch = train_batched.shuffle(buffer_size=train_batch_size)
