#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from libs import utils
print('Import Done')
mnist = utils.download_data()

print('=== Data ===')
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
print('Train Img   {:12}, {:20}'.format(trainimg.shape, type(trainimg)))
print('Train Label {:12}, {:20}'.format(trainlabel.shape, type(trainlabel)))
print('Test  Img   {:12}, {:20}'.format(testimg.shape, type(testimg)))
print('Test  Label {:12}, {:20}'.format(testlabel.shape, type(testlabel)))

idx = 0
# utils.show_image(trainimg[idx, :], trainlabel[idx, :], 'Train {}'.format(idx))
