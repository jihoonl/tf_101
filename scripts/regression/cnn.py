#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from libs import utils

if __name__ == '__main__':
    mnist = utils.download_data(input_data)
    data = {}
    data['train'] = mnist.train
    data['test'] = {
        'x': mnist.test.images,
        'y': mnist.test.labels
    }

    device_type = '/gpu:0'

    n_input = 784
    n_output = 10
