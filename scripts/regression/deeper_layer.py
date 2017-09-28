#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from libs import utils


def create_network(n_features):
    # Input and Output
    x = tf.placeholder('float', [None, n_features[0]])
    y = tf.placeholder('float', [None, n_features[-1]])
    dropout_keep_prop = tf.placeholder('float')

    # Variables
    stddev = 0.05
    weights = [
        tf.Variable(tf.random_normal([a, b], stddev=stddev))
        for a, b in zip(n_features[:-1], n_features[1:])
    ]
    biases = [tf.Variable(tf.random_normal([a])) for a in n_features[1:]]
    print('Length W: {}, B: {}'.format(len(weights), len(biases)))
    return x, y, weights, biases, dropout_keep_prop


def multilayer_perceptron(X, weights, biases, keep_prop):
    l = []
    l.append(tf.nn.relu(tf.add(tf.matmul(X, weights[0]), biases[0])))

    for i, (w, b) in enumerate(zip(weights[1:-1], biases[1:-1])):
        l.append(tf.nn.relu(tf.add(tf.matmul(l[i], w), b)))
    l.append(
        tf.matmul(tf.nn.dropout(l[-1], keep_prop), weights[-1]) + biases[-1])
    return (l[-1])


if __name__ == '__main__':

    data = utils.get_mnist_data()

    # Network Topologies
    n_features = []
    n_features.append(784)  # Input size. Image(28 x 28)
    n_features.append(512)  # Layer 1
    n_features.append(512)  # Layer 2
    n_features.append(256)  # Layer 3
    n_features.append(10)  # Output. number of classes

    x, y, weights, biases, dropout_keep_prob = create_network(n_features)

    # Prediction
    pred = multilayer_perceptron(x, weights, biases, dropout_keep_prob)

    # Loss and optimizer
    cost, optm, corr, accr = utils.get_functions(pred, y)

    # Parameters
    training_epochs = 20
    batch_size = 100
    display_step = 4

    additional_inputs = {}
    additional_inputs[dropout_keep_prob] = [0.6, 1.0]

    utils.run(x, y, data, pred, cost, optm, corr, accr, training_epochs,
              batch_size, display_step, additional_inputs)
