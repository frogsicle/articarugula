import tensorflow as tf
import numpy as np

# input from outside of TensorFlow
# we're learning an XOR gate
x_numpy = np.array([[0., 1.],
                    [0., 0.],
                    [1., 0.],
                    [1., 1.]])

y_numpy = np.array([[1.],
                    [0.],
                    [1.],
                    [0.]])

## setting up graph (just input & labels)
# not the way to _do_ it, but a decent way to understand what's done
# where the input data comes in
x_input = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y_input = tf.placeholder(dtype=tf.float32, shape=[None,1])

# trainables input -> hidden
x_bias = tf.Variable(tf.random_normal([9]), trainable=True)
weights_01 = tf.Variable(tf.random_normal([2, 9]), trainable=True)

# hidden layer & its non linearity
hidden = tf.nn.xw_plus_b(x_input, weights_01, x_bias)
hidden_nonlin = tf.maximum(0., hidden)

# trainables hidden -> output
hidden_bias = tf.Variable(tf.random_normal([1]), trainable=True)
weights_02 = tf.Variable(tf.random_normal([9, 1]), trainable=True)

# predictions & its non linearity
pre_predictions = tf.nn.xw_plus_b(hidden_nonlin, weights_02, hidden_bias)
predictions = tf.sigmoid(pre_predictions)

# scoring and training
loss = tf.losses.sigmoid_cross_entropy(logits=pre_predictions,
                                      multi_class_labels=y_input)

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# initializing variables
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # this is the part that would run on the graphics card
    sess.run(init_op)

    for i in range(10000):
        _, loss_as_number = sess.run([train_op, loss],
                                      feed_dict={x_input: x_numpy,
                                                 y_input: y_numpy})
        if i % 500 == 0:
            print('loss:', loss_as_number)

    out = sess.run(predictions, feed_dict={x_input: x_numpy, y_input: y_numpy})

print('predictions:\n{}'.format(out))
