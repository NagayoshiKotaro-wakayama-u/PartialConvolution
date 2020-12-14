from sklearn.cluster import KMeans
from argparse import ArgumentParser
import pickle
import numpy as np
import pdb
import os
import matplotlib.pylab as plt
import tensorflow as tf
import copy


def parse_args():
    parser = ArgumentParser(description='Generate toy data for Inpainting')
    parser.add_argument('-dspath','--dataSetPath',type=str,default="gaussianToyData",help='Name of dataSet (default=gaussianToyData)')

    return  parser.parse_args()
 
#-----------------
# create spatial histogram using convolution with the kernel of all one-value
def compSpatialHist(x,kSize=64,sSize=4,isNormMode='sum',thre=0.05):
    # binarize images
    x_bin = x>thre
    #x_bin = tf.constant(np.expand_dims(x_bin,-1), dtype=tf.float32)
    x_bin = tf.constant(x_bin, dtype=tf.float32)

    # kernel with all ones
    kernel=np.ones([kSize,kSize,1,1])
    kernel = tf.constant(kernel, dtype=tf.float32)

    # histogram using conv2d
    x_conv = tf.nn.conv2d(x_bin,kernel,strides=[1,sSize,sSize,1],padding='VALID')
    shape = tf.shape(x_conv)
    x_conv_flat = tf.reshape(x_conv,[shape[0],shape[1]*shape[2]])

    if isNormMode == 'max':
        x_conv_flat = x_conv_flat/tf.reduce_max(x_conv_flat,axis=1,keepdims=1)
    elif isNormMode == 'sum':
        x_conv_flat = x_conv_flat/tf.reduce_sum(x_conv_flat,axis=1,keepdims=1)

    x_conv = tf.reshape(x_conv_flat,[shape[0],shape[1],shape[2]])

    return x_conv
#-----------------

#-----------------
def compKL(p1,p2,smallV=1e-10):
    shape = p1.shape
    p1_reshape = tf.reshape(p1,[shape[0],-1])
    p2_reshape = tf.reshape(p2,[shape[0],-1])

    kl = tf.reduce_sum(p1_reshape*(tf.math.log(p1_reshape+smallV) - tf.math.log(p2_reshape+smallV)),axis=1)

    return kl
#-----------------

if __name__ == "__main__":
    args = parse_args()

    shape = (512,512) # 画像のサイズ
    plotNum = 20

    keys = ["train","valid","test"]
    dsPath = "data" + os.sep + args.dataSetPath # データセットのパス
    if not os.path.isdir(dsPath):
        print(f"folder {dsPath} does not exist.")

    # set paths
    imagePath = [dsPath + os.sep + k + ".pickle" for k in keys]
    maskPath = [dsPath + os.sep+k+"_mask.pickle" for k in keys]

    #-----------------
    # load data
    # gt images
    with open(imagePath[0],'rb') as fp:
        x_gt = pickle.load(fp)

    # mask
    with open(maskPath[0],'rb') as fp:
        mask = pickle.load(fp)


    # create mask images
    x = x_gt['images']
    x_mask = copy.deepcopy(x)
    x_mask[mask==0]=0
    #-----------------

    # create spatial histogram
    x_hist = compSpatialHist(x, kSize=128, sSize=4, thre=0.05)
    x_mask_hist = compSpatialHist(x_mask, kSize=128, sSize=4, thre=0.05)

    # compute kl distance
    kl = compKL(x_mask_hist,x_hist)

    pdb.set_trace()

    #-----------------
    # plot
    for i in range(plotNum):
        fig = plt.figure()
        ax=fig.add_subplot(2,2,1)
        ax.imshow(x[i,:,:,0],vmin=0,vmax=255,cmap="jet")
        ax.set_title('dense')

        ax=fig.add_subplot(2,2,2)
        im = ax.imshow(x_mask[i,:,:,0],vmin=0,vmax=255,cmap="jet")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        ax.set_title('sparse')

        ax=fig.add_subplot(2,2,3)
        im = ax.imshow(x_hist[i],cmap="jet")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        ax.set_title('hist of dense')

        ax=fig.add_subplot(2,2,4)
        im = ax.imshow(x_mask_hist[i],cmap="jet")
        ax.set_title(f'hist of sparse, kl={kl[i]:.4f}')
        fig.colorbar(im, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()
    #-----------------
    


