import tensorflow as tf
from numpy.random import RandomState

def get_weight(shape, lbd):
    var = tf.Variable(tf.random_normal(shape=shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lbd)(var))
    return var


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf. placeholder(tf.float32, shape=(None, 1))

batch_size =8

layer_dimension = [2, 10, 10, 10, 1]
n_layers = len(layer_dimension)

cur_layer = x

in_dimension = layer_dimension[0]

for i in range(1, n_layers):
    out_dimension = layer_dimension[i]
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    in_dimension = layer_dimension[i]

mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

tf.add_to_collection('losses', mse_loss)

loss = tf.add_n(tf.get_collection('losses'))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for x1, x2 in X]

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # print(sess.run(w1))
    # print(sess.run(w2))

    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        if i % 1000 == 0:
            total_cross_entropy = sess.run(loss, feed_dict={x: X[start:end], y_: Y[start:end]})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    # print(sess.run(w1))
    # print(sess.run(w2))
