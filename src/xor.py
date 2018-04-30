import numpy as np
import tensorflow as tf


def draw_batch(batch_size):
    xs = np.random.choice([0.0, 1.0], [batch_size, 2])
    ys = np.array(list(map(lambda x: [x[0] != x[1]], xs)))
    return (xs, ys)


# first layer - weights + bias
W1 = tf.Variable(tf.random_uniform([2, 2], -1, 1), name="W1")
b1 = tf.Variable(tf.zeros([2]), name="b1")

# second layer - weights + bias
W2 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="W2")
b2 = tf.Variable(tf.zeros([1]), name="b2")

# input vector, layer outputs and label
x = tf.placeholder(tf.float32, [64, 2], "x")
h1 = tf.sigmoid(tf.matmul(x, W1) + b1)
h2 = tf.sigmoid(tf.matmul(h1, W2) + b2)
y = tf.placeholder(tf.float32, [64, 1], "y")

# network loss and optimizer
loss = tf.losses.mean_squared_error(y, h2)
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(10000):
    (xs, ys) = draw_batch(64)
    (loss_value, _) = sess.run([loss, optimizer], {x: xs, y: ys})

    if i % 100 == 0:
        print(loss_value)