#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from libs import utils


def conv_simple(inputs, W, B):
    # reshape input
    i = tf.reshape(inputs, shape=[-1, 28, 28, 1])

    # convolution
    conv1 = tf.nn.conv2d(i, W['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    # Add bias
    conv2 = tf.nn.bias_add(conv1, B['bc1'])
    # Pass relu
    conv3 = tf.nn.relu(conv2)
    # Max pooling
    pool = tf.nn.max_pool(
        conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Vectorize
    dense = tf.reshape(pool, [-1, W['wd1'].get_shape().as_list()[0]])
    # fully connected layer
    out = tf.add(tf.matmul(dense, W['wd1']), B['bd1'])
    # Return
    return {
        'input_r': i,
        'conv1': conv1,
        'conv2': conv2,
        'conv3': conv3,
        'pool': pool,
        'dense': dense,
        'out': out
    }


if __name__ == '__main__':
    data = utils.get_mnist_data()

    device_type = '/gpu:0'

    with tf.device(device_type):
        n_input = 784
        n_output = 10
        weights = {
            'wc1':
            tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),
            'wd1':
            tf.Variable(tf.random_normal([14 * 14 * 64, n_output], stddev=0.1))
        }
        biases = {
            'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
            'bd1': tf.Variable(tf.random_normal([n_output], stddev=0.1))
        }

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_output])

    # Parameter
    learning_rate = 0.001
    training_epochs = 10
    batch_size = 100
    display_step = 1

    with tf.device(device_type):
        pred = conv_simple(x, weights, biases)['out']
        cost, optm, corr, accr = utils.get_functions(pred, learning_rate)

    # Saver
    save_step = 1
    save_dirs = utils.save_path('mnist')
    saver = tf.train.Saver(max_to_keep=3)
