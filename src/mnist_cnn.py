import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# input layer
# shape = [batch size, image 1st dim, image 2nd dim, single pixel value]
x = tf.placeholder(tf.float32, [64, 28, 28, 1], "x")

# first convolution layer using 32 filters of size 5x5 to extract features
# same = zero padding, stride = (1, 1)
conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[5, 5], paddings="same", activation=tf.nn.relu)

# first pooling layer with kernel size 2x2 and stride=2 for all dimensions
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# second convolution layer now with 64 filters
conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

# second pooling layer
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# TODO fully connected layers
# TODO loss + optimizer
# TODO training
# TODO test

