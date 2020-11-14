import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

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


# mnist_data = tf.keras.datasets.mnist.load_data()
# (train_images, train_labels), (test_images, test_labels) = mnist_data

for ex in tfds.load('genomics_ood', split='train'):
  print(ex)
