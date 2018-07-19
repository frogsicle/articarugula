import tensorflow as tf
import numpy as np

# input from outside of TensorFlow
x_numpy = np.array([[0., 1.],
                    [0., 0.],
                    [1., 0.],
                    [1., 1.]])
y_numpy = np.array([[0.], [0.], [0.], [1.]])
print(x_numpy.shape)


## super basic for sake of understanding
#x_data = tf.placeholder(dtype=tf.float32, shape=[None,6])
#y_data = tf.placeholder(dtype=tf.float32, shape=[None,1], name='meh')
#
#
#x_bias = tf.Variable(tf.random_normal([5]), trainable=True)
#weights_01 = tf.Variable(tf.random_normal([6,5]), trainable=True)
#
#hidden = tf.nn.xw_plus_b(x_data, weights_01, x_bias)
#hidden_nonlin = tf.maximum(0., hidden)
#
#hidden_bias = tf.Variable(tf.random_normal([1]), trainable=True)
#weights_02 = tf.Variable(tf.random_normal([5, 1]), trainable=True)
#
#pre_predictions = tf.nn.xw_plus_b(hidden_nonlin, weights_02, hidden_bias)
#pre_predictions = tf.reshape(pre_predictions, [1,1])
#predictions = tf.sigmoid(pre_predictions)

# we're building a graph
#x_input = tf.Variable([0., 1., 2.])
x_input = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y_input = tf.placeholder(dtype=tf.float32, shape=[None,1])


hidden = tf.layers.dense(x_input, 5)
hidden_nonlinear = tf.nn.relu(hidden)

pre_predictions = tf.layers.dense(hidden_nonlinear, 1)
predictions = tf.sigmoid(pre_predictions)

loss = tf.losses.sigmoid_cross_entropy(logits=pre_predictions,
                                      multi_class_labels=y_input)


train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# prep to be able to save our trained model
saver = tf.train.Saver()

init_op = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init_op)
    for i in range(1000):
        _, loss_as_number = sess.run([train_op, loss],
                                      feed_dict={x_input: x_numpy,
                                                 y_input: y_numpy})
        if i % 100 == 0:
            print('loss:', loss_as_number)
    # actually save
    saver.save(sess, 'save_here/a_checkpoint')


with tf.Session() as sess:
    saver.restore(sess, 'save_here/a_checkpoint')
    pred = sess.run(predictions, feed_dict={
        x_input: [[0, -1],
                  [1.2, 0.],
                  [1, 1]]})

    print('a prediction:', pred)
    print(tf.trainable_variables())
    some_weights = tf.get_default_graph(
    ).get_tensor_by_name('dense/kernel:0')
    number_weights = sess.run(some_weights)

x = tf.trainable_variables()
for item in x:
    print(item)