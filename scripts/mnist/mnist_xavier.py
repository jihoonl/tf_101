#!/usr/bin/env python
'''
Deeper Multilayer perceptron with XAVIER Init
Xavier init from Project: https://github.com/aymericdamien/TensorFlow-Examples/
@Sungjoon Choi (sungjoon.choi@cps.lab.snu.ac.kr)
'''
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
    weights = [
        tf.get_variable(
            str(i), shape=[a, b], initializer=utils.xavier_init(a, b))
        for i, (a, b) in enumerate(zip(n_features[:-1], n_features[1:]))
    ]
    biases = [tf.Variable(tf.random_normal([a])) for a in n_features[1:]]
    print('Length W: {}, B: {}'.format(len(weights), len(biases)))
    return x, y, weights, biases, dropout_keep_prop


def multilayer_perceptron(X, weights, biases, keep_prop):
    l = []
    l.append(
        tf.nn.dropout(
            tf.nn.relu(tf.add(tf.matmul(X, weights[0]), biases[0])), keep_prop))
    for i, (w, b) in enumerate(zip(weights[1:-1], biases[1:-1])):
        l.append(
            tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(l[i], w), b)), keep_prop))
    l.append(tf.matmul(l[-1], weights[-1]) + biases[-1])
    return (l[-1])


if __name__ == '__main__':

    data = utils.get_mnist_data()

    # Network Topoligies
    n_features = []
    n_features.append(784)  # Input size. Image(28 x 28)
    n_features.append(256)  # Layer 1
    n_features.append(256)  # Layer 2
    n_features.append(256)  # Layer 3
    n_features.append(256)  # Layer 4
    n_features.append(10)  # Output. number of classes

    x, y, weights, biases, dropout_keep_prob = create_network(n_features)

    # Prediction
    pred = multilayer_perceptron(x, weights, biases, dropout_keep_prob)

    # Loss and optimizer
    cost, optm, corr, accr = utils.get_functions(pred, y)

    # Parameters
    learning_rate = 0.001
    training_epochs = 50
    batch_size = 100
    display_step = 1

    additional_inputs = {}
    additional_inputs[dropout_keep_prob] = [0.7, 1.0]

    utils.run(x, y, data, pred, cost, optm, corr, accr, training_epochs,
              batch_size, display_step, additional_inputs)
