# do PCA first
# then do k-maens

from PCV.tools import imtools

import pickle

from scipy import *

from pylab import *

from PIL import Image

from scipy.cluster.vq import *

from PCV.tools import pca

imlist = imtools.get_imlist('/home/constantine/PycharmProjects/tf_playground/clustering/data/a_selected_thumbs/')

im = array(Image.open(imlist[0]))  # open one image to get the size

m, n = im.shape[:2]  # get the size of the images
imnbr = len(imlist)  # get the number of images
print("The number of images is %d" % imnbr)

# Create matrix to store all flattened images
immatrix = array([array(Image.open(imname)).flatten() for imname in imlist], 'f')

# PCA降维
V, S, immean = pca.pca(immatrix)

# 保存均值和主成分
#f = open('./a_pca_modes.pkl', 'wb')
f = open('./a_pca_modes.pkl', 'wb')
pickle.dump(immean, f)
pickle.dump(V, f)
f.close()

# get list of images
imlist = imtools.get_imlist('/home/constantine/PycharmProjects/tf_playground/clustering/data/a_selected_thumbs/')
imnbr = len(imlist)

# load model file
with open('./a_pca_modes.pkl', 'rb') as f:
    immean = pickle.load(f)
    V = pickle.load(f)

# create matrix to store all flattened images
immatrix = array([array(Image.open(im)).flatten() for im in imlist], 'f')

# project on the 40 first PCs
immean = immean.flatten()
projected = array([dot(V[:16], immatrix[i]-immean) for i in range(imnbr)])

# k-means
projected = whiten(projected)
centroids, distortion = kmeans(projected, 4)
code, distance = vq(projected, centroids)

# plot clusters
for k in range(4):
    ind = where(code == k)[0]
    figure()
    gray()
    for i in range(minimum(len(ind), 40)):
        subplot(4, 10, i+1)
        imshow(immatrix[ind[i]].reshape((25, 25)))
        axis('off')
show()
