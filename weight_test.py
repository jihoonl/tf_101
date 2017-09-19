#!/usr/bin/env python

import numpy as np
import tensorflow as tf

import utils

print("Package Loaded")

if __name__ == '__main__':
    print('')
    print('Initialized')
    sess = tf.Session()

    # Variable
    print('== Variable ==')
    # print('=== Input ===')
    weight = tf.Variable(tf.ones([2, 1]))
    # utils.print_tf(weight)
    # print('=== Output ===')
    init = tf.initialize_all_variables()
    sess.run(init)
    weight_out = sess.run(weight)
    utils.print_tf(weight_out)

    # Placeholder
    print('== Placeholder ==')
    x = tf.placeholder(tf.float32, [None, 2])
    utils.print_tf(x)
    print('== Mat Mul ==')
    matmuls = tf.matmul(x, weight)
    utils.print_tf(matmuls)
    data = np.ones([10, 2])
    print(data)
    # print data
    oper_out = sess.run(matmuls, feed_dict={x: data})
    utils.print_tf(oper_out)
