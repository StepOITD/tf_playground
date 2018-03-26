import numpy as np
from pylab import *
from scipy.cluster.vq import *

def calculate_acc(code, label, step, CLUSTER_NUMBER):
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
