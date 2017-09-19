#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

print('Import Done')

print('Download and Extract Mnist dataset')
mnist = input_data.read_data_sets('data/', one_hot=True)
print
print('Type of \'mnist\' is {}'.format(type(mnist)))
print('# of train data is {}'.format(mnist.train.num_examples))
print('# of test  data is {}'.format(mnist.test.num_examples))
