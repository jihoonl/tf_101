
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


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

def download_data():
    mnist = input_data.read_data_sets('data/', one_hot=True)

    if os.path.exists('data'):
        return mnist
    print('Download and Extract Mnist dataset')
    print
    print('Type of \'mnist\' is {}'.format(type(mnist)))
    print('# of train data is {}'.format(mnist.train.num_examples))
    print('# of test  data is {}'.format(mnist.test.num_examples))
    return mnist
