#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def f(x, a, b):
    n = x.size
    vals = np.zeros((1, n))
    for i in range(0, n):
        ax = np.multiply(a, x.item(i))
        val = np.add(ax, b)
        vals[0, i] = val
    return vals


if __name__ == '__main__':

    # Generate Training data
    wref = 0.5
    bref = -1
    n = 20
    var = 0.001
    np.random.seed(1)
    train_x = np.random.random((1, n))
    ref_y = f(train_x, wref, bref)
    train_y = ref_y + np.sqrt(var) * np.random.randn(1, n)

    # Plot
    plt.figure(1)
    plt.plot(train_x[0, :], ref_y[0, :], 'ro', label='Original')
    plt.plot(train_x[0, :], train_y[0, :], 'bo', label='Training')
    plt.axis('equal')
    plt.legend(loc='lower right')
    # plt.show()

    # Parameters
    epochs = 2000
    display_step = 50

    # Set tensorflow graph
    X = tf.placeholder(tf.float32, name='input')
    Y = tf.placeholder(tf.float32, name='output')
    W = tf.Variable(np.random.randn(), name='weight')
    b = tf.Variable(np.random.randn(), name='bias')

    # Construct a model
    activation = tf.add(tf.multiply(X, W), b)

    # Define error measure and optimizer
    learning_rate = 0.01
    cost = tf.reduce_mean(tf.pow(activation - Y, 2))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    threshold = 8e-4

    # initializer
    init = tf.initialize_all_variables()

    # run
    with tf.Session() as sess:
        sess.run(init)
        for e in range(epochs):
            for x, y in zip(train_x[0, :], train_y[0, :]):
                sess.run(optimizer, feed_dict={X:x, Y:y})

            # Check cost
            costval = sess.run(cost, feed_dict={X: train_x, Y:train_y})
            print('Epoch: {:4d}, cost= {:.5f}'.format(e + 1, costval))
            wtemp = sess.run(W)
            btemp = sess.run(b)
            print('    Wtemp is {:.4f}, Btemp is {:.4f}'.format(wtemp, btemp))
            print('    Wref is {:.4f}, bref is {:.4f}'.format(wref, bref))
            if costval < threshold:
                break
        wopt = sess.run(W)
        bopt = sess.run(b)
        fopt = f(train_x, wopt, bopt)

    # Plot Results
    plt.figure(2)
    plt.plot(train_x[0, :], ref_y[0, :], 'ro', label='Original data')
    plt.plot(train_x[0, :], train_y[0, :], 'bo', label='Training data')
    plt.plot(train_x[0, :], fopt[0, :], 'ko', label='Fitted Line')
    plt.axis('equal')
    plt.legend(loc='lower right')
    plt.show()
