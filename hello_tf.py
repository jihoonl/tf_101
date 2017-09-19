#!/usr/bin/env python

import numpy as np
import tensorflow as tf

import utils

print("Package Loaded")


def test_data(data):
    if not isinstance(data, list):
        data = [data]

    input_data = []
    output_data = []
    for d in data:
        input_data.append(tf.constant(d))

    for i in input_data:
        output_data.append(sess.run(i))

    print('==== Input  ====')
    utils.print_tf(input_data)

    print('==== Output ====')
    utils.print_tf(output_data)

    return input_data, output_data


if __name__ == '__main__':
    print('')
    print('Initialized')
    sess = tf.Session()

    # Test String
    # test_data('Hello, Tensorflow')

    # Test float
    output_data, input_data = test_data([1.5, 2.5])

    # Add
    print('== Add ==')
    sums = tf.add(*input_data)
    utils.print_tf(sums)
    sums_out = sess.run(sums)
    utils.print_tf(sums_out)

    # Multiplication
    print('== Mul ==')
    muls = tf.mul(*input_data)
    muls_out = sess.run(muls)
    utils.print_tf(muls_out)
