import tensorflow as tf
from pylab import *
from scipy.cluster.vq import *
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

FEATURE_DIM = 24
CLUSTER_NUMBER = 20
LEARNING_RATE_BASE = 1e-5
LEARNING_RATE_DECAY = 0.99

def calculate_acc(code, label, step):
    code = np.array(code)
    label = np.array(label)
    true_set = list(set(label))
    # print(len(code))
    # print(len(label))
    summ = 0
    for i in range(CLUSTER_NUMBER):
        part = np.array(label[where(code == i)])
        str = "| "
        ma = -1
        f = -1
        max_count = 0
        for j in true_set:
            if len(part) != 0:
                a = sum(part == j)
                b = len(part)
                per = a/b
                if per > ma:
                    ma = per
                    f = j
                    max_count = a
                temp = "%d: %.2g" % (j, per)
            else:
                temp = "%d: None" % (j)
            str += temp + "| "
        summ += max_count
        if step % 1000 == 0:
            print("in cluster %d:" % i)
            print(str + "|| MAX PERCENT: %d: %g of %d" % (f, ma, len(part)))
    print("TOTAL ACC %g" % (summ/len(label)))


def gen_matrix(code, centroids):
    matrix = [centroids[i] for i in code]
    return np.array(matrix)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # strides[0]: number of images
    # strides[1]: height of images
    # strides[2]: width of images
    # strides[3]: channel of images
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def head(mnist):

    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    global_step = tf.Variable(0, trainable=False)

    W_conv1 = weight_variable([5, 5, 1, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 16, 20])
    b_conv2 = bias_variable([20])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 20, 128])
    b_fc1 = bias_variable([128])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 20])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([128, FEATURE_DIM])
    b_fc2 = bias_variable([FEATURE_DIM])
    features = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

    centroid_matrix = tf.placeholder(dtype=tf.float32, shape=[None, FEATURE_DIM])

    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(features-centroid_matrix, 2), 1)))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               20000,
                                               LEARNING_RATE_DECAY)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)


    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        xs = mnist.test.images
        ys = mnist.test.labels
        codebooks = []
        for i in range(20000):
            extract_features = sess.run(features, feed_dict={x: xs, keep_prob: 1.0})
            centroids, variance = kmeans(extract_features, CLUSTER_NUMBER)
            code, distance = vq(extract_features, centroids)
            cent_matrix = gen_matrix(code, centroids)
            # print(i)
            if i % 10 == 0:
                codebooks.append(code)
                loss_value = sess.run(loss, feed_dict={x: xs, keep_prob: 1.0, centroid_matrix: cent_matrix})
                print("At %d step(s), the loss is %g" % (i, loss_value))
                calculate_acc(code, ys, i)

                # name = input()

            sess.run(train_op, feed_dict={x: xs, keep_prob: 1.0, centroid_matrix: cent_matrix})

        return codebooks


mnist = input_data.read_data_sets("mnist_data", one_hot=False)
pre = head(mnist)


# import tensorflow as tf
# with tf.Session() as sess:
#     v = tf.constant([1, 2, 3, 4], dtype=tf.float32)
#     print(sess.run(tf.sqrt(v)))