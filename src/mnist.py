import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

W1 = tf.Variable(tf.random_uniform([784, 396], -1, 1), name="W1")
b1 = tf.Variable(tf.zeros([396]), name="b1")

W2 = tf.Variable(tf.random_uniform([396, 10], -1, 1), name="W2")
b2 = tf.Variable(tf.zeros([10]), name="b2")

x = tf.placeholder(tf.float32, [64, 784], "x")
h1 = tf.sigmoid(tf.matmul(x, W1) + b1)
h2 = tf.nn.softmax(tf.matmul(h1, W2) + b2)
y = tf.placeholder(tf.float32, [64, 10], "y")

loss = tf.losses.mean_squared_error(y, h2)
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

mnist = input_data.read_data_sets('../data/mnist_data', one_hot=True)

for i in range(10000):
    (xs, ys) = mnist.train.next_batch(64)
    (loss_value, _) = sess.run([loss, optimizer], {x:xs, y:ys})

    if i % 1000 == 0:
        print(loss_value)

(xs, ys) = mnist.train.next_batch(64)
pred = sess.run(h2, {x:xs})

for (p, y) in zip(pred, ys):
    print("prediction: %d, label: %d" % (np.argmax(p), np.argmax(y)))
