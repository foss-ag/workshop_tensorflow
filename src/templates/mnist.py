import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# TODO: network + loss + optimizer
#
# your code here
#

# start Tensorflow session and initialize variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# get MNIST data set
mnist = input_data.read_data_sets('../data/mnist_data', one_hot=True)

# TODO: run training
for i in range(10000):
    #
    # your code here
    #
    pass
