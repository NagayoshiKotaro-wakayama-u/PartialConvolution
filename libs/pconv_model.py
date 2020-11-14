import os
import sys
import numpy as np
from datetime import datetime
import pdb

import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation, Lambda
from keras.layers.merge import Concatenate
from keras import backend as K
from keras.utils.multi_gpu_utils import multi_gpu_model

from libs.pconv_layer import PConv2D
from PIL import Image

def calcDet(lis,dim):
    if not(dim ==2 or dim==3):
        print("Can't calculate Determinant ( calculate only dimension 2 or 3 )")
        return None

    if dim==2:
        det = lis[0][0]*lis[1][1] - lis[0][1]*lis[1][0]
    elif dim==3:
        det = lis[0][0]*(lis[1][1]*lis[2][2] - lis[1][2]*lis[2][1]) - lis[0][1]*(lis[1][0]*lis[2][2]-lis[1][2]*lis[2][0]) + lis[0][2]*(lis[1][0]*lis[2][1]-lis[1][1]*lis[2][0])

    return det


class PConvUnet(object):

    def __init__(self, img_rows=512, img_cols=512, inference_only=False, net_name='default', gpus=1, KLthre=0.1, isUsedKL=True, exist_point_file=""):
        """Create the PConvUnet. If variable image size, set img_rows and img_cols to None

        Args:
            img_rows (int): image height.
            img_cols (int): image width.
            inference_only (bool): initialize BN layers for inference.
            net_name (str): Name of this network (used in logging).
            gpus (int): How many GPUs to use for training.
            KLthre (float): threshold of KL-loss.
            exist_point_file (str): 存在する点が１・その他が０である入力と同サイズの画像のパス（入力画像内に欠損部以外の未観測点がある場合に使用）
        """

        # Settings
        self.img_rows = img_rows
        self.img_cols = img_cols
        #self.img_overlap = 30
        self.inference_only = inference_only
        self.net_name = net_name
        self.gpus = gpus
        self.losses = None
        self.KLthre = KLthre
        self.isUsedKL = isUsedKL

        # X座標,Y座標の行列
        self.Xmap = K.constant(np.tile([[i for i in range(self.img_cols)]],(self.img_rows,1))[np.newaxis,:,:,np.newaxis])
        self.Ymap = K.constant(np.tile(np.array([[i for i in range(self.img_rows)]]).T, (1,self.img_cols))[np.newaxis,:,:,np.newaxis])

        # 存在する点が１・その他が０である入力と同サイズの画像を設定
        if exist_point_file=="":
            self.exist = K.constant(np.ones([self.img_rows,self.img_cols,1]))
        else:
            self.exist = K.constant(np.array(Image.open(exist_point_file))[np.newaxis,:,:,np.newaxis]/255)

        # Assertions
        assert self.img_rows >= 256, 'Height must be >256 pixels'
        assert self.img_cols >= 256, 'Width must be >256 pixels'

        # Set current epoch
        self.current_epoch = 0

        # Create UNet-like model
        if self.gpus <= 1:
            self.model, inputs_mask = self.build_pconv_unet()
            self.compile_pconv_unet(self.model, inputs_mask)
        else:
            with tf.device("/cpu:0"):
                self.model, inputs_mask = self.build_pconv_unet()
            self.model = multi_gpu_model(self.model, gpus=self.gpus)
            self.compile_pconv_unet(self.model, inputs_mask)

    def build_pconv_unet(self, train_bn=True):

        # INPUTS
        inputs_img = Input((self.img_rows, self.img_cols, 1), name='inputs_img')
        inputs_mask = Input((self.img_rows, self.img_cols, 1), name='inputs_mask')

        # ENCODER
        def encoder_layer(img_in, mask_in, filters, kernel_size, bn=True):
            conv, mask = PConv2D(filters, kernel_size, strides=2, padding='same')([img_in, mask_in])
            if bn:
                conv = BatchNormalization(name='EncBN'+str(encoder_layer.counter))(conv, training=train_bn)
            conv = Activation('relu')(conv)
            encoder_layer.counter += 1
            return conv, mask
        encoder_layer.counter = 0

        e_conv1, e_mask1 = encoder_layer(inputs_img, inputs_mask, 64, 7, bn=False)
        e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 128, 5)
        e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 256, 5)
        e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 512, 3)
        e_conv5, e_mask5 = encoder_layer(e_conv4, e_mask4, 512, 3)

        # DECODER
        def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel_size, bn=True):
            up_img = UpSampling2D(size=(2,2))(img_in)
            up_mask = UpSampling2D(size=(2,2))(mask_in)
            concat_img = Concatenate(axis=3)([e_conv,up_img])
            concat_mask = Concatenate(axis=3)([e_mask,up_mask])
            conv, mask = PConv2D(filters, kernel_size, padding='same')([concat_img, concat_mask])
            if bn:
                conv = BatchNormalization()(conv)
            conv = LeakyReLU(alpha=0.2)(conv)
            return conv, mask

        d_conv6, d_mask6 = decoder_layer(e_conv5, e_mask5, e_conv4, e_mask4, 512, 3)
        d_conv7, d_mask7 = decoder_layer(d_conv6, d_mask6, e_conv3, e_mask3, 256, 3)
        d_conv8, d_mask8 = decoder_layer(d_conv7, d_mask7, e_conv2, e_mask2, 128, 3)
        d_conv9, d_mask9 = decoder_layer(d_conv8, d_mask8, e_conv1, e_mask1, 64, 3)
        d_conv10, _ = decoder_layer(d_conv9, d_mask9, inputs_img, inputs_mask, 3, 3, bn=False)
        outputs = Conv2D(1, 1, activation = 'sigmoid', name='outputs_img')(d_conv10)

        # Setup the model inputs / outputs
        model = Model(inputs=[inputs_img, inputs_mask], outputs=outputs)

        return model, inputs_mask

    def compile_pconv_unet(self, model, inputs_mask, lr=0.0002):
        model.compile(
            optimizer = Adam(lr=lr),
            loss=self.loss_total(inputs_mask),
            metrics=[self.PSNR,self.loss_KL]
        )

    def loss_total(self, mask):
        """
        Creates a loss function which sums all the loss components
        and multiplies by their weights. See paper eq. 7.
        """
        def loss(y_true, y_pred):

            # Compute predicted image with non-hole pixels set to ground truth
            y_comp = mask * y_true + (1-mask) * y_pred

            # Compute loss components
            l1 = self.loss_valid(mask, y_true, y_pred)
            l2 = self.loss_hole(mask, y_true, y_pred)
            l3 = self.loss_tv(mask, y_comp)
            l4 = self.loss_KL(y_true, y_pred)

            # Return loss function
            if self.isUsedKL:
                return l1 + 6*l2 + 0.1*l3 + l4
            else:
                return l1 + 6*l2 + 0.1*l3

        return loss

    
    def loss_KL(self,y_true, y_pred,dim=2):
        thre = self.KLthre
        # y_predの中でthre以上の値の座標を取り出すためのマスクを作成
        pred = y_pred*self.exist # shape=[N,512,512,1]
        pred = tf.nn.relu(pred - thre)
        pred = tf.math.sign(pred)
        # マスクを座標にかけてthre以上の値のXY座標を取得・平均を計算
        X1 = pred*self.Xmap
        Y1 = pred*self.Ymap
        num1 = tf.reduce_sum(pred) + 10e-6
        muX1 = tf.reduce_sum(X1,axis=[1,2],keep_dims=True)/num1 # shape=[N,1,1,1] 
        muY1 = tf.reduce_sum(Y1,axis=[1,2],keep_dims=True)/num1

        # y_trueの中でthre以上の値の座標を取り出すためのマスクを作成
        true = y_true*self.exist
        true = tf.nn.relu(true - thre)
        true = tf.math.sign(true)
        # 上記と同様
        X2 = true*self.Xmap
        Y2 = true*self.Ymap
        num2 = tf.reduce_sum(true) + 10e-6
        muX2 = tf.reduce_sum(X2,axis=[1,2],keep_dims=True)/num2
        muY2 = tf.reduce_sum(Y2,axis=[1,2],keep_dims=True)/num2

        # 分散共分散行列
        disX1 = tf.abs((X1-muX1)*pred)
        disY1 = tf.abs((Y1-muY1)*pred)
        cov1 = [
            [tf.reduce_sum(disX1**2,axis=[1,2,3])/num1, tf.reduce_sum(disX1*disY1,axis=[1,2,3])/num1],
            [tf.reduce_sum(disX1*disY1,axis=[1,2,3])/num1, tf.reduce_sum((disY1**2),axis=[1,2,3])/num1]
        ]

        disX2 = tf.abs((X2-muX2)*true)
        disY2 = tf.abs((Y2-muY2)*true)
        cov2 = [
            [tf.reduce_sum((disX2**2),axis=[1,2,3])/num2, tf.reduce_sum(disX2*disY2,axis=[1,2,3])/num2],
            [tf.reduce_sum(disX2*disY2,axis=[1,2,3])/num2, tf.reduce_sum((disY2**2),axis=[1,2,3])/num2]
        ]

        # 多変量正規分布(2変量)のKL-Divergenceを計算
        # 第一項 : shape=[N]
        det1 = calcDet(cov1,dim)
        det2 = calcDet(cov2,dim)

        # 第二項 : shape=[N]
        tr21 = (cov1[0][0]*cov2[1][1] - 2*cov1[0][1]*cov2[0][1] + cov1[1][1]*cov2[0][0])/(det2+1e-10)

        # 第三項 : shape=[N]
        d_mu = [tf.squeeze(muX1-muX2,axis=[1,2,3]), tf.squeeze(muY1-muY2,axis=[1,2,3])]
        sq = ((d_mu[0]**2)*cov2[1][1] - 2*d_mu[0]*d_mu[1]*cov2[0][1] + (d_mu[1]**2)*cov2[0][0] )/(det2+1e-10)

        KL = 0.5*(tf.log(det2/(det1+1e-10)) + tr21 + sq -dim)

        return KL


    def loss_hole(self, mask, y_true, y_pred):
        """Pixel L1 loss within the hole / mask"""
        return self.l1((1-mask) * y_true, (1-mask) * y_pred)

    def loss_valid(self, mask, y_true, y_pred):
        """Pixel L1 loss outside the hole / mask"""
        return self.l1(mask * y_true, mask * y_pred)

    def loss_tv(self, mask, y_comp):
        """Total variation loss, used for smoothing the hole region, see. eq. 6"""

        # Create dilated hole region using a 3x3 kernel of all 1s.
        kernel = K.ones(shape=(3, 3, mask.shape[3], mask.shape[3]))
        dilated_mask = K.conv2d(1-mask, kernel, data_format='channels_last', padding='same')

        # Cast values to be [0., 1.], and compute dilated hole region of y_comp
        dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
        P = dilated_mask * y_comp

        # Calculate total variation loss
        a = self.l1(P[:,1:,:,:], P[:,:-1,:,:])
        b = self.l1(P[:,:,1:,:], P[:,:,:-1,:])
        return a+b

    def fit_generator(self, generator, *args, **kwargs):
        """Fit the U-Net to a (images, targets) generator

        Args:
            generator (generator): generator supplying input image & mask, as well as targets.
            *args: arguments to be passed to fit_generator
            **kwargs: keyword arguments to be passed to fit_generator
        """
        self.model.fit_generator(
            generator,
            *args, **kwargs
        )

    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())

    def load(self, filepath, train_bn=True, lr=0.0002):

        # Create UNet-like model
        self.model, inputs_mask = self.build_pconv_unet(train_bn)
        self.compile_pconv_unet(self.model, inputs_mask, lr)

        # Load weights into model
        epoch = int(os.path.basename(filepath).split('.')[1].split('-')[0])
        assert epoch > 0, "Could not parse weight file. Should include the epoch"
        self.current_epoch = epoch
        self.model.load_weights(filepath)

    @staticmethod
    def PSNR(y_true, y_pred):
        """
        PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        The equation is:
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)

        Our input is scaled with be within the range -2.11 to 2.64 (imagenet value scaling). We use the difference between these
        two values (4.75) as MAX_I
        """
        #return 20 * K.log(4.75) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
        return - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

    @staticmethod
    def current_timestamp():
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    @staticmethod
    def l1(y_true, y_pred):
        """Calculate the L1 loss used in all loss calculations"""
        if K.ndim(y_true) == 4:
            return K.mean(K.abs(y_pred - y_true), axis=[1,2,3])
        elif K.ndim(y_true) == 3:
            return K.mean(K.abs(y_pred - y_true), axis=[1,2])
        else:
            raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")

    @staticmethod
    def gram_matrix(x, norm_by_channels=False):
        """Calculate gram matrix used in style loss"""

        # Assertions on input
        assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
        assert K.image_data_format() == 'channels_last', "Please use channels-last format"

        # Permute channels and get resulting shape
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        shape = K.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]

        # Reshape x and do batch dot product
        features = K.reshape(x, K.stack([B, C, H*W]))
        gram = K.batch_dot(features, features, axes=2)

        # Normalize with channels, height and width
        gram = gram /  K.cast(C * H * W, x.dtype)

        return gram

    # Prediction functions
    ######################
    def predict(self, sample, **kwargs):
        """Run prediction using this model"""
        return self.model.predict(sample, **kwargs)
