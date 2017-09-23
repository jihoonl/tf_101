#!/usr/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from libs import utils

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import matplotlib.pyplot as plt


if __name__ == '__main__':
    mnist = utils.download_data(input_data)
    train_x = mnist.train.images
    train_y = mnist.train.labels
    test_x = mnist.test.images
    test_y = mnist.test.images

    x = tf.placeholder('float', [None, 784])
    y = tf.placeholder('float', [None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Logistics regression
    activation = tf.nn.softmax(tf.matmul(x, W) + b)

    # Cost function
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(activation), axis=1))

    # Optimizer
    learning_rate = 0.01
    optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Prediction
    pred = tf.cast(tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1)), 'float')
    # Accuracy
    accr = tf.reduce_mean(pred)
    # Initializer
    init = tf.global_variables_initializer()

    t_epochs = 10
    batch_size = 100
    display_step = 5
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(t_epochs):
            avg_cost = 0.
            num_batch = int(mnist.train.num_examples / batch_size)
            for i in range(num_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
                feeds = {x: batch_xs, y: batch_ys}
                avg_cost += sess.run(cost, feed_dict=feeds) / num_batch

            # Display
            if epoch % display_step == 0:
                feeds_train = {x: batch_xs, y: batch_ys}
                feeds_test = {x: mnist.test.images, y: mnist.test.labels}
                train_acc = sess.run(accr, feed_dict=feeds_train)
                test_acc = sess.run(accr, feed_dict=feeds_test)
                print(
                    'Epoch : {:03d}/{:03d}, Cost: {:.9f}, Train Acc: {:.3f}, Test Acc: {:.3f}'.
                    format(epoch, t_epochs, avg_cost, train_acc, test_acc))
        train_pred = sess.run(pred, feed_dict={x: train_x, y: train_y})
        test_pred = sess.run(pred, feed_dict=feeds_test)
        test_accr = sess.run(accr, feed_dict=feeds_test)
        print(len(test_pred))
        print(sum(test_pred))
        print(test_accr)

print('Done')
