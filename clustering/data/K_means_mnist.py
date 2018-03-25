from PCV.tools import imtools

import pickle

from scipy import *

from pylab import *

from PIL import Image

from scipy.cluster.vq import *

from PCV.tools import pca

from tensorflow.examples.tutorials.mnist import input_data

def calculate_acc(code, label, CLUSTER_NUMBER, step=0):
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


mnist = input_data.read_data_sets("mnist_data", one_hot=False)

xs = mnist.test.images
ys = mnist.test.labels

V, S, immean = pca.pca(xs)

f = open('./mnist_test_pca_modes.pkl', 'wb')
pickle.dump(immean, f)
pickle.dump(V, f)
f.close()

immean = immean.flatten()
imnbr = 10000
projected = array([dot(V[:32], xs[i]-immean) for i in range(imnbr)])


# k-means
projected = whiten(projected)

centroids, distortion = kmeans(projected, 15)
code, distance = vq(projected, centroids)

calculate_acc(code, ys, 10)

