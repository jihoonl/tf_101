#!/usr/bin/env python

import tensorflow as tf
# import matplotlib.pyplot as plt

from libs import utils


def multilayer_perceptron(X, W, B):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, W['h1']), B['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, W['h2']), B['b2']))
    return tf.matmul(layer_2, W['out']) + B['out']


if __name__ == '__main__':
    data = utils.get_mnist_data()

    # Network topology
    n_hidden_1 = 256
    n_hidden_2 = 128
    n_input = 784
    n_classes = 10

    # Inputs and outputs
    x = tf.placeholder('float', [None, n_input])
    y = tf.placeholder('float', [None, n_classes])

    # Network parameters
    stddev = 0.1
    weights = {
        'h1':
        tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
        'h2':
        tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
        'out':
        tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=stddev))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    print('Network Ready')

    # Prediction
    pred = multilayer_perceptron(x, weights, biases)

    # Loss and optimizer
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    # Optimizer
    rate = 0.1
    optm = tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(cost)
    # optm = tf.train.AdamOptimizer(learning_rate=rate).minimize(cost)
    corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accr = tf.reduce_mean(tf.cast(corr, 'float'))
    print('Functions Ready')

    # parameters
    training_epochs = 50
    batch_size = 100
    display_step = 1

    utils.run(
        x,
        y,
        data,
        pred,
        cost,
        optm,
        corr,
        accr,
        training_epochs=training_epochs,
        batch_size=batch_size,
        display_step=display_step)
