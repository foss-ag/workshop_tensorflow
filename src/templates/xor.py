import numpy as np
import tensorflow as tf


def draw_batch(batch_size):
    xs = np.random.choice([0.0, 1.0], [batch_size, 2])
    ys = np.array(list(map(lambda x: [x[0] != x[1]], xs)))
    return (xs, ys)


# TODO: first layer - weights + bias
#
# your code here
#

# TODO: second layer - weights + bias
#
# your code here
#

# TODO: input vector, layer outputs and label
#
# your code here
#

# TODO: network loss and optimizer
#
# your code here
#

# start Tensorflow session and initialize variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# TODO: run training
for i in range(10000):
    #
    # your code here
    #
    pass
