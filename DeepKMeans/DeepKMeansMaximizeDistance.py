import tensorflow as tf
from pylab import *
from scipy.cluster.vq import *
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PCV.tools import imtools
import pickle
from scipy import *
from pylab import *
from PIL import Image
from scipy.cluster.vq import *
from PCV.tools import pca

import DeepKMeans.tools as tools

from tensorflow.examples.tutorials.mnist import input_data

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


def inference(mnist):

    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    global_step = tf.Variable(0, trainable=False)

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    W_conv4 = weight_variable([3, 3, 128, 256])
    b_conv4 = bias_variable([256])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)

    h_pool4_flat = tf.reshape(h_pool2, [-1, 2*2*256])

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        xs = mnist.test.images
        ys = mnist.test.labels
        extract_features = sess.run(h_pool4_flat, feed_dict={x: xs})
        print(extract_features.shape)
        V, S, immean = pca.pca(extract_features)
        immean = immean.flatten()
        imnbr = 10000
        projected = array([dot(V[:32], extract_features[i] - immean) for i in range(imnbr)])
        centroids, distortion = kmeans(projected, 15)

        code, distance = vq(projected, centroids)

        tools.calculate_acc(code, ys, 10)

    return extract_features

mnist = input_data.read_data_sets("mnist_data", one_hot=False)
pre = inference(mnist)

