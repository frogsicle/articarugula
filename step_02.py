import tensorflow as tf
import numpy as np

# input from outside of TensorFlow

x_numpy = np.array([[0., 1.],
                    [0., 0.],
                    [1., 0.],
                    [1., 1.]])

y_numpy = np.array([[1.], [0.], [1.], [0.]])

# setting up graph (just input & labels)
# where the input data comes in
x_input = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y_input = tf.placeholder(dtype=tf.float32, shape=[None,1])

# initializing variables
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # this is the part that would run on the graphics card
    sess.run(init_op)
    x_out, y_out = sess.run([x_input, y_input], feed_dict={x_input: x_numpy, y_input: y_numpy})

print('x_out:\n{}'.format(x_out))
print('y_out:\n{}'.format(y_out))