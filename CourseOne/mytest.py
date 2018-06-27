import pylab
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

dir = "./testImgs/"
filename = dir + "la_defense.jpg"
ori_image = np.array(ndimage.imread(filename, flatten=False))
dim1_pad = np.pad(ori_image, ((50, 50), (0, 0), (0, 0)), 'constant', constant_values=0)
dim2_pad = np.pad(ori_image, ((0, 0), (50, 50), (0, 0)), 'maximum')
dim3_pad = np.pad(ori_image, ((50, 50), (50, 50), (0, 0)), 'constant', constant_values=200)
print("origin shape:" + str(ori_image.shape))
print("vertical pad:" + str(dim1_pad.shape))
print("horizontal pad:" + str(dim2_pad.shape))
print("all pad:" + str(dim3_pad.shape))

fig, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(ori_image)
axarr[0, 1].imshow(dim1_pad[:, :, 0])
axarr[1, 0].imshow(dim2_pad[:, :, 1])
axarr[1, 1].imshow(dim3_pad[:, :, 2])
pylab.show()
