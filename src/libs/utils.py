
import os
import numpy as np
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
