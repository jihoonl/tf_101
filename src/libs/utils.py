
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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


def download_data(input_data, debug=False):
    datapath = os.environ.get('DATAPATH', 'data/')
    data = input_data.read_data_sets(datapath, one_hot=True)

    if debug:
        print('Download and Extract data dataset')
        print
        print('Type of \'data\' is {}'.format(type(data)))
        print('# of train data is {}'.format(data.train.num_examples))
        print('# of test  data is {}'.format(data.test.num_examples))
    return data


def run(x, y, data, pred, cost, optm, corr, accr,
        training_epochs=20, batch_size=100, display_step=4):
    # Initialize
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(data['train'].num_examples / batch_size)

            # Iteration
            for i in range(total_batch):
                batch_xs, batch_ys = data['train'].next_batch(batch_size)
                feeds = {x: batch_xs, y: batch_ys}
                sess.run(optm, feed_dict=feeds)
                avg_cost += sess.run(cost, feed_dict=feeds)
            avg_cost = avg_cost / total_batch

            # Display
            if (epoch) % display_step == 0:
                print('Epoch: {:03d}/{:03d}, cost: {:.9f}'.format(
                    epoch, training_epochs, avg_cost))
                feeds = {x: batch_xs, y: batch_ys}
                train_acc = sess.run(accr, feed_dict=feeds)
                print('- Train Accuracy: {:.3f}'.format(train_acc))
                feeds = {x: data['test']['x'], y: data['test']['y']}
                test_acc = sess.run(accr, feed_dict=feeds)
                print('- Test Accuracy: {:.3f}'.format(test_acc))

        print('Epoch: {:03d}/{:03d}, cost: {:.9f}'.format(
            epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys}
        train_acc = sess.run(accr, feed_dict=feeds)
        print('- Train Accuracy: {:.3f}'.format(train_acc))
        feeds = {x: data['test']['x'], y: data['test']['y']}
        test_acc = sess.run(accr, feed_dict=feeds)
        print('- Test Accuracy: {:.3f}'.format(test_acc))
