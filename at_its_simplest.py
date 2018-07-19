import tensorflow as tf
import numpy as np

# input from outside of TensorFlow
x_numpy = np.array([[0., 1.],
                    [0., 0.],
                    [1., 0.],
                    [1., 1.]])
y_numpy = np.array([[0.], [0.], [0.], [1.]])
print(x_numpy.shape)

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

    some_weights = tf.get_default_graph(
    ).get_tensor_by_name('find_me/kernel:0')
    number_weights = sess.run(some_weights)

x = tf.trainable_variables()
for item in x:
    print(item)