import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data


def print_tf(xx):
    if not isinstance(xx, list):
        xx = [xx]

    for x in xx:
        print('Type  :\n{}'.format(type(x)))
        print('Value :\n{}'.format(x))


def show_image(data, label, name):
    img = np.reshape(data, (28, 28))
    l = np.argmax(label)
    plt.matshow(img, cmap=plt.get_cmap('gray'))
    t = '{} -> {}'.format(name, l)
    plt.title(t)
    plt.show()
    print(t)


def _get_datapath():
    return os.environ.get('DATAPATH', 'data/')


def save_path(postfix):
    datapath = _get_datapath()
    return os.path.join(datapath, postfix)


def download_data(input_data, debug=False):
    datapath = _get_datapath() 
    data = input_data.read_data_sets(datapath, one_hot=True)

    if debug:
        print('Download and Extract data dataset')
        print
        print('Type of \'data\' is {}'.format(type(data)))
        print('# of train data is {}'.format(data.train.num_examples))
        print('# of test  data is {}'.format(data.test.num_examples))
    return data


def get_mnist_data():
    mnist = download_data(mnist_input_data)
    data = {}
    data['train'] = mnist.train
    data['test'] = {}
    data['test']['x'] = mnist.test.images
    data['test']['y'] = mnist.test.labels
    return data


def xavier_init(n_inputs, n_outputs, uniform=True):
    """
    Set the parameter initialization using the method prescribed.
    This method is designed to keep the scale of the gradients roughly the same in all layers

    Xavier Glorot and Yoshua Bengio (2010):
        Understanding the difficulty of training deep feedforward
        neural networks. International conference on artificial
        intelligence and statistics

    Args:
        n_inputs: # of input nodes into each output.
        n_outputs: # of output nodes for each input.
        uniform: If true, use a uniform distribution,
                 otherwise use a normal.
    Returns:
        An initializer.
    """
    if uniform:
        initialization_constant = 6.0
        init_range = tf.sqrt(initialization_constant / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3.0 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean
        initialization_constant = 3.0
        stddev = tf.sqrt(initialization_constant / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def get_functions(pred, y, learning_rate=0.001):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accr = tf.reduce_mean(tf.cast(corr, 'float'))
    return cost, optm, corr, accr


def create_feeds(x, xx, y, yy, addition, index):
    feeds = {x: xx, y: yy}
    if addition:
        for k, v in addition.items():
            feeds[k] = v[index]
    return feeds


def run(x,
        y,
        data,
        pred,
        cost,
        optm,
        corr,
        accr,
        training_epochs=20,
        batch_size=100,
        display_step=4,
        additional_inputs=None):
    # Initialize
    init = tf.global_variables_initializer()
    print('-- Start --')

    with tf.Session() as sess:
        sess.run(init)

        for e in range(training_epochs):
            epoch = e + 1

            avg_cost = 0.
            total_batch = int(data['train'].num_examples / batch_size)

            # Iteration
            for i in range(total_batch):
                batch_xs, batch_ys = data['train'].next_batch(batch_size)
                feeds = create_feeds(x, batch_xs, y, batch_ys,
                                     additional_inputs, 0)
                sess.run(optm, feed_dict=feeds)
                feeds = create_feeds(x, batch_xs, y, batch_ys,
                                     additional_inputs, 1)
                avg_cost += sess.run(cost, feed_dict=feeds)
            avg_cost = avg_cost / total_batch

            # Display
            if (epoch) % display_step == 0:
                print('Epoch: {:03d}/{:03d}, cost: {:.9f}'.format(
                    epoch, training_epochs, avg_cost))
                feeds = create_feeds(x, batch_xs, y, batch_ys,
                                     additional_inputs, 1)
                train_acc = sess.run(accr, feed_dict=feeds)
                print('- Train Accuracy: {:.3f}'.format(train_acc))
                feeds = create_feeds(x, data['test']['x'], y, data['test']['y'],
                                     additional_inputs, 1)
                test_acc = sess.run(accr, feed_dict=feeds)
                print('- Test Accuracy: {:.3f}'.format(test_acc))

        print('Epoch: {:03d}/{:03d}, cost: {:.9f}'.format(
            epoch, training_epochs, avg_cost))
        feeds = create_feeds(x, batch_xs, y, batch_ys, additional_inputs, 1)
        train_acc = sess.run(accr, feed_dict=feeds)
        print('- Train Accuracy: {:.3f}'.format(train_acc))
        feeds = create_feeds(x, data['test']['x'], y, data['test']['y'],
                             additional_inputs, 1)
        test_acc = sess.run(accr, feed_dict=feeds)
        print('- Test Accuracy: {:.3f}'.format(test_acc))
