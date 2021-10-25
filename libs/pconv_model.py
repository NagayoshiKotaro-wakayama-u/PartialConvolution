import os
import sys
import numpy as np
from datetime import datetime
import pdb
from copy import deepcopy
import cv2

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, Lambda, Multiply, Dense, Flatten, GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras import backend as K
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback,Callback
from keras_tqdm import TQDMCallback

from libs.pconv_layer import PConv2D, siteConv,Encoder,Decoder,sitePConv,siteEncoder,siteDecoder,PKEncoder,siteNormalize
# from libs.createSpatialHistogram import compSpatialHist,compKL
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

    def __init__(self, img_rows=512, img_cols=512, lr=0.0002,loss_weights=[1,6,0.1], inference_only=False, net_name='default', gpus=1, thre=0.2, KLthre=0.1, histKLthre=0.05,
     isUsedKL=True, isUsedHistKL=True, isUsedLLH=True,LLHonly=False,existOff=False,exist_point_file="", 
    histFSize=64,histSSize=4, truefirst= False, predfirst=False,  KLbias=True, KLonly=False,maskGaus=None):

        # Settings
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.learning_rate = lr
        self.loss_weights=loss_weights
        #self.img_overlap = 30
        self.inference_only = inference_only
        self.net_name = net_name
        self.gpus = gpus
        self.losses = None
        self.thre = thre
        self.KLthre = KLthre
        self.histKLthre = histKLthre
        self.maskGaus = maskGaus

        # KLの第一引数をpredとtrueのどちらにするか
        self.truefirst = truefirst
        self.predfirst = predfirst
        # KLにバイアスをかけるか
        self.KLbias = KLbias
        # KLのみで学習を行うか
        self.KLonly = KLonly
        self.LLHonly= LLHonly

        self.isUsedKL = True if KLonly else isUsedKL
        self.isUsedHistKL = isUsedHistKL
        self.isUsedLLH = True if LLHonly else isUsedLLH

        self.existOff = existOff
        self.histFSize = histFSize
        self.histSSize = histSSize

        # X座標,Y座標の行列
        self.Xmap = K.constant(np.tile([[i for i in range(self.img_cols)]],(self.img_rows,1))[np.newaxis,:,:,np.newaxis])
        self.Ymap = K.constant(np.tile(np.array([[i for i in range(self.img_rows)]]).T, (1,self.img_cols))[np.newaxis,:,:,np.newaxis])

        # 存在する点が１・その他が０である入力と同サイズの画像を設定
        if exist_point_file=="":
            self.exist_img = np.ones([self.img_rows,self.img_cols,1])
        else:
            self.exist_img = np.array(Image.open(exist_point_file))[np.newaxis,:,:,np.newaxis]/255
            if self.maskGaus is not None:
                # ガウシアンフィルタによって平滑化
                kernel = (maskGaus,maskGaus)
                self.expand_exist_img = cv2.GaussianBlur(self.exist_img[0,:,:,0]*255,kernel,0)
                _,self.expand_exist_img = cv2.threshold(self.expand_exist_img, 1, 255, cv2.THRESH_BINARY)
                self.expand_exist_img = self.expand_exist_img[np.newaxis,:,:,np.newaxis]/255
        
        self.exist = K.constant(self.exist_img)
        self.obsNum = np.sum(self.exist_img)

        # Assertions
        # assert self.img_rows >= 256, 'Height must be >256 pixels'
        # assert self.img_cols >= 256, 'Width must be >256 pixels'

        # Set current epoch
        self.current_epoch = 0

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # INPUTS
        self.inputs_img = Input((self.img_rows, self.img_cols, 1), name='inputs_img')
        self.inputs_mask = Input((self.img_rows, self.img_cols, 1), name='inputs_mask')
        # self.true_vec = Input((5),name='true_vec')

        # decide model
        self.encoder1 = Encoder(64, 7, 1, bn=False)
        self.encoder2 = Encoder(128,5, 2)
        self.encoder3 = Encoder(256,5, 3)
        self.encoder4 = Encoder(512,3, 4) #TODO:元に戻す(512,3,4)
        self.encoder5 = Encoder(512,3, 5) #TODO:元に戻す(512,3,5)
        
        self.decoder6 = Decoder(512, 3)
        self.decoder7 = Decoder(256,3)
        self.decoder8 = Decoder(128,3)
        self.decoder9 = Decoder(64,3)
        self.decoder10 = Decoder(3,3,bn=False)
        self.conv2d = Conv2D(1,1,activation='sigmoid',name='output_img')
        self.flatten = Flatten()
        self.density = Dense(5,name="output_vec")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.ones33 = K.ones(shape=(3, 3, 1, 1))

        # Create UNet-like model
        if self.gpus <= 1:
            # self.outputs_img,self.outputs_vec = self.build_pconv_unet()
            self.outputs_img = self.build_pconv_unet()
            # self.model = Model(inputs=[self.inputs_img, self.inputs_mask], outputs=[self.outputs_img,self.outputs_vec])
            self.model = Model(inputs=[self.inputs_img, self.inputs_mask],outputs=self.outputs_img)
            self.compile_pconv_unet(self.model, self.inputs_mask,lr=self.learning_rate)
            # self.loss = loss_total(self.inputs_mask)(self.)
        else:
            with tf.device("/cpu:0"):
                self.model = Model(inputs=[self.inputs_img, self.inputs_mask], outputs=self.build_pconv_unet())
            self.model = multi_gpu_model(self.model, gpus=self.gpus)
            self.compile_pconv_unet(self.model, self.inputs_mask,lr=self.learning_rate)

    def build_pconv_unet(self, train_bn=True):
        e_conv1, e_mask1 = self.encoder1(self.inputs_img,self.inputs_mask)
        e_conv2, e_mask2 = self.encoder2(e_conv1,e_conv1)
        e_conv3, e_mask3 = self.encoder3(e_conv2,e_conv2)
        e_conv4, e_mask4 = self.encoder4(e_conv3,e_conv3)
        e_conv5, e_mask5 = self.encoder5(e_conv4,e_conv4)

        d_conv6, d_mask6 = self.decoder6(e_conv5, e_mask5, e_conv4, e_mask4)
        d_conv7, d_mask7 = self.decoder7(d_conv6, d_mask6, e_conv3, e_mask3)
        d_conv8, d_mask8 = self.decoder8(d_conv7, d_mask7, e_conv2, e_mask2)
        d_conv9, d_mask9 = self.decoder9(d_conv8, d_mask8, e_conv1, e_mask1)
        d_conv10, _ = self.decoder10(d_conv9, d_mask9, self.inputs_img, self.inputs_mask)

        outputs = self.conv2d(d_conv10)
        
        return outputs

    def compile_pconv_unet(self, model, inputs_mask, lr=0.0002):
        model.compile(
            optimizer = Adam(lr=lr),
            loss= self.loss_total(inputs_mask),
            metrics=[
                self.loss_total(inputs_mask),
                self.loss_origin(inputs_mask),
                self.PSNR
            ]
        )

    def loss_total(self, mask):
        """
        Creates a loss function which sums all the loss components
        and multiplies by their weights. See paper eq. 7.
        """
        def lossFunction(y_true, y_pred):
            # Compute predicted image with non-hole pixels set to ground truth
            # Compute loss components

            origin = self.loss_origin(mask)(y_true,y_pred)

            # Return loss function
            return origin
        return lossFunction

    # partialConvolution 自体の損失関数
    def loss_origin(self,mask):
        def original(y_true,y_pred):

            # 観測値部分の誤差
            l1 = self.loss_valid(mask, y_true, y_pred)

            # e_mask ＝ 欠損部（陸地）：０　観測点：１　海域部：１（海洋部を損失に加えない）
            # mask = 0,1,0   exist_img = 1,1,0   1-exist_img = 0,0,1
            if self.existOff:
                if self.maskGaus is None:
                    e_mask = mask+(1-self.exist_img)
                else:
                    e_mask = mask+(1-self.expand_exist_img)
            else:
                e_mask = mask

            # 欠損部の誤差
            l2 = self.loss_hole(e_mask, y_true, y_pred)
            
            # 欠損部のまわり1pxの誤差
            y_comp = e_mask * y_true + (1-e_mask) * y_pred
            l3 = self.loss_tv(e_mask, y_comp)

            w1,w2,w3 = self.loss_weights

            res = w1*l1 + w2*l2

            return tf.add(res,w3*l3,name="loss_origin")

        return original

    def loss_hole(self, mask, y_true, y_pred):
        """Pixel L1 loss within the hole / mask"""
        return self.l1((1-mask) * y_true, (1-mask) * y_pred)

    def loss_valid(self, mask, y_true, y_pred):
        """Pixel L1 loss outside the hole / mask"""
        return self.l1(mask * y_true, mask * y_pred)

    def loss_tv(self, mask, y_comp):
        """Total variation loss, used for smoothing the hole region, see. eq. 6"""

        # Create dilated hole region using a 3x3 kernel of all 1s.
        kernel = self.ones33
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
        # Load weights into model
        epoch = os.path.basename(filepath).split('.')[1].split('-')[0]
        try:
            epoch = int(epoch)
        except ValueError:
            self.current_epoch = 100
        else:
            self.current_epoch = epoch

        self.model.load_weights(filepath)

    # @staticmethod
    def PSNR(self,y_true, y_pred):
        pred = y_pred*self.exist_img
        #return 20 * K.log(4.75) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
        return - 10.0 * K.log(K.mean(K.square(pred - y_true))) / K.log(10.0)

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


class sitePConvUnet(object):

    def __init__(self, img_rows=512, img_cols=512, lr=0.0002, use_site=False, inference_only=False, net_name='default', gpus=1,
     existOff=False,exist_point_file="",  posEmbChan=1, maskGaus=None):
        # Settings
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.learning_rate = lr
        #self.img_overlap = 30
        self.inference_only = inference_only
        self.net_name = net_name
        self.gpus = gpus
        self.losses = None
        self.use_site = use_site
        self.maskGaus = maskGaus

        self.existOff = existOff

        # 存在する点が１・その他が０である入力と同サイズの画像を設定
        if exist_point_file=="":
            exist_img = np.ones([self.img_rows,self.img_cols,1])
        else:
            exist_img = np.array(Image.open(exist_point_file))[np.newaxis,:,:,np.newaxis]/255
            if self.maskGaus is not None:
                # ガウシアンフィルタによって平滑化
                kernel = (maskGaus,maskGaus)
                self.expand_exist_img = cv2.GaussianBlur(self.exist_img[0,:,:,0]*255,kernel,0)
                _,self.expand_exist_img = cv2.threshold(self.expand_exist_img, 1, 255, cv2.THRESH_BINARY)
                self.expand_exist_img = self.expand_exist_img[np.newaxis,:,:,np.newaxis]/255

        self.exist = K.constant(exist_img)
        self.obsNum = np.sum(exist_img)

        # Assertions
        # assert self.img_rows >= 256, 'Height must be >256 pixels'
        # assert self.img_cols >= 256, 'Width must be >256 pixels'

        # Set current epoch
        self.current_epoch = 0

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # INPUTS
        self.inputs_img = Input((self.img_rows, self.img_cols, 1), name='inputs_img')
        self.inputs_mask = Input((self.img_rows, self.img_cols, 1), name='inputs_mask')
        if use_site:
            self.inputs_site = Input((int(self.img_rows/4), int(self.img_cols/4), 1), name='inputs_site')
            inputs = [self.inputs_img, self.inputs_mask,self.inputs_site]
        else:
            inputs = [self.inputs_img, self.inputs_mask]

        # decide model
        self.encoder1 = Encoder(64, 7, 1, bn=False)
        self.encoder2 = Encoder(128,5, 2)
        # pdb.set_trace()
        if use_site:
            self.site_encoder3 = siteEncoder(256,5, 3)
            self.encoder4 = Encoder(512,3, 4)
            self.encoder5 = Encoder(512,3, 5)
        else:
            self.encoder3 = Encoder(256,5,3,posEmbChan=posEmbChan,use_bias=True)
            self.encoder4 = Encoder(512,3, 4,posEmbChan=posEmbChan,use_bias=True)
            self.encoder5 = Encoder(512,3, 5,posEmbChan=posEmbChan,use_bias=True)
        
        
        self.decoder6 = Decoder(512, 3)
        self.decoder7 = Decoder(256,3)
        self.decoder8 = Decoder(128,3)
        self.decoder9 = Decoder(64,3)
        self.decoder10 = Decoder(3,3,bn=False)
        self.conv2d = Conv2D(1,1,activation='sigmoid',name='output_img')
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.ones33 = K.ones(shape=(3, 3, 1, 1))

        # Create UNet-like model
        if self.gpus <= 1:
            self.outputs_img = self.build_pconv_unet()
            self.model = Model(inputs=inputs,outputs=self.outputs_img)
            self.compile_pconv_unet(self.model, self.inputs_mask, lr=self.learning_rate)
        else:
            with tf.device("/cpu:0"):
                self.model = Model(inputs=inputs, outputs=self.build_pconv_unet())
            self.model = multi_gpu_model(self.model, gpus=self.gpus)
            self.compile_pconv_unet(self.model, self.inputs_mask,lr=self.learning_rate)

    def build_pconv_unet(self, train_bn=True):
        e_conv1, e_mask1 = self.encoder1(self.inputs_img,self.inputs_mask)
        e_conv2, e_mask2 = self.encoder2(e_conv1,e_conv1)
        if self.use_site:
            e_conv3, e_mask3 = self.site_encoder3(e_conv2,e_conv2,self.inputs_site)
        else:
            e_conv3, e_mask3 = self.encoder3(e_conv2,e_conv2)
        e_conv4, e_mask4 = self.encoder4(e_conv3,e_conv3)
        e_conv5, e_mask5 = self.encoder5(e_conv4,e_conv4)

        d_conv6, d_mask6 = self.decoder6(e_conv5, e_mask5, e_conv4, e_mask4)
        d_conv7, d_mask7 = self.decoder7(d_conv6, d_mask6, e_conv3, e_mask3)
        d_conv8, d_mask8 = self.decoder8(d_conv7, d_mask7, e_conv2, e_mask2)
        d_conv9, d_mask9 = self.decoder9(d_conv8, d_mask8, e_conv1, e_mask1)
        d_conv10, _ = self.decoder10(d_conv9, d_mask9, self.inputs_img, self.inputs_mask)

        outputs = self.conv2d(d_conv10)
        
        return outputs

    def compile_pconv_unet(self, model, inputs_mask, lr=0.0002):
        model.compile(
            optimizer = Adam(lr=lr),
            loss= self.loss_total(inputs_mask),
            metrics=[
                self.loss_total(inputs_mask),
                self.loss_origin(inputs_mask),
                self.PSNR
            ]
        )

    def loss_total(self, mask):
        """
        Creates a loss function which sums all the loss components
        and multiplies by their weights. See paper eq. 7.
        """
        def lossFunction(y_true, y_pred):
            # Compute predicted image with non-hole pixels set to ground truth
            # Compute loss components
            origin = self.loss_origin(mask)(y_true,y_pred)

            # Return loss function
            return origin
        return lossFunction

    # partialConvolution 自体の損失関数
    def loss_origin(self,mask):
        def original(y_true,y_pred):

            # 観測値部分の誤差
            l1 = self.loss_valid(mask, y_true, y_pred)

            # e_mask ＝ 欠損部（陸地）：０　観測点：１　海域部：１（海洋部を損失に加えない）
            # mask = 0,1,0   exist_img = 1,1,0   1-exist_img = 0,0,1
            if self.existOff:
                if self.maskGaus is None:
                    e_mask = mask+(1-self.exist_img)
                else:
                    e_mask = mask+(1-self.expand_exist_img)
            else:
                e_mask = mask

            # 欠損部の誤差
            l2 = self.loss_hole(e_mask, y_true, y_pred)
            
            # 欠損部のまわり1pxの誤差
            y_comp = e_mask * y_true + (1-e_mask) * y_pred
            l3 = self.loss_tv(e_mask, y_comp)

            res = l1+6*l2

            return tf.add(res,0.1*l3,name="loss_origin")
            
        return original

    def loss_hole(self, mask, y_true, y_pred):
        """Pixel L1 loss within the hole / mask"""
        return self.l1((1-mask) * y_true, (1-mask) * y_pred)

    def loss_valid(self, mask, y_true, y_pred):
        """Pixel L1 loss outside the hole / mask"""
        return self.l1(mask * y_true, mask * y_pred)

    def loss_tv(self, mask, y_comp):
        """Total variation loss, used for smoothing the hole region, see. eq. 6"""

        # Create dilated hole region using a 3x3 kernel of all 1s.
        kernel = self.ones33
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
        # Load weights into model
        epoch = os.path.basename(filepath).split('.')[1].split('-')[0]
        try:
            epoch = int(epoch)
        except ValueError:
            self.current_epoch = 100
        else:
            self.current_epoch = epoch

        self.model.load_weights(filepath)

    # @staticmethod
    def PSNR(self,y_true, y_pred):
        pred = y_pred*self.exist_img
        #return 20 * K.log(4.75) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
        return - 10.0 * K.log(K.mean(K.square(pred - y_true))) / K.log(10.0)

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
 

class PKConvUnet(object):

    def __init__(self, img_rows=512, img_cols=512, lr=0.0002, loss_weights=[1,6,0.1], use_site=True, inference_only=False, net_name='default', gpus=1,
     existOff=False,exist_point_file="", posEmbChan=1,opeType="add",PKConvlayer=[3,4,5],
     encFNum=[64,128,256,512,512],sCNNFNum=[8,8,8,8,8],eachChannel=False,useSiteCNN=False,sCNNBias=False,sCNNActivation=None
     ,sCNNSinglePath=False,useSiteNormalize=False,maskGaus=None,sConvKernelLearn=False,sConvChan=None,site_range=[0,1],sklSigmoid=False
     ,learnMultiSiteW=False):
        # Settings
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.learning_rate = lr
        self.lossWeights = loss_weights
        #self.img_overlap = 30
        self.inference_only = inference_only
        self.net_name = net_name
        self.gpus = gpus
        self.losses = None
        self.use_site = use_site # 既存のサイト特性を使用するか否か
        self.opeType = opeType
        self.posEmbChan = posEmbChan
        self.firstLayer = PKConvlayer[0]
        self.existOff = existOff
        self.useSiteCNN = useSiteCNN
        self.useSiteNormalize = useSiteNormalize
        self.sCNNBias = sCNNBias
        self.sCNNSinglePath = sCNNSinglePath
        self.maskGaus = maskGaus
        self.sConvKernelLearn = sConvKernelLearn
        self.sConvChan = sConvChan
        self.learnMultiSiteW = learnMultiSiteW
        
        # 存在する点が１・その他が０である入力と同サイズの画像を設定
        if exist_point_file=="":
            self.exist_img = np.ones([self.img_rows,self.img_cols,1])
        else:
            self.exist_img = np.array(Image.open(exist_point_file))[np.newaxis,:,:,np.newaxis]/255
            if self.maskGaus is not None:
                # ガウシアンフィルタによって平滑化
                kernel = (maskGaus,maskGaus)
                self.expand_exist_img = cv2.GaussianBlur(self.exist_img[0,:,:,0]*255,kernel,0)
                _,self.expand_exist_img = cv2.threshold(self.expand_exist_img, 1, 255, cv2.THRESH_BINARY)
                self.expand_exist_img = self.expand_exist_img[np.newaxis,:,:,np.newaxis]/255
        
        self.exist = K.constant(self.exist_img)
        self.obsNum = np.sum(self.exist_img)

        # Set current epoch
        self.current_epoch = 0

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # INPUTS
        self.inputs_img = Input((self.img_rows, self.img_cols, 1), name='inputs_img')
        self.inputs_mask = Input((self.img_rows, self.img_cols, 1), name='inputs_mask')

        if use_site:# 位置特性を初めの入力に用いているかどうか
            devide = 2**(self.firstLayer-1)
            site_row = int(self.img_rows/devide)
            site_col = int(self.img_cols/devide)
            self.inputs_site = Input((site_row, site_col, posEmbChan), name='inputs_site')
            self.inputs = [self.inputs_img, self.inputs_mask,self.inputs_site]
        elif useSiteCNN:
            self.inputs_site = Input((self.img_rows, self.img_cols, posEmbChan), name='inputs_site')
            self.inputs = [self.inputs_img, self.inputs_mask,self.inputs_site]
        else:
            self.inputs = [self.inputs_img, self.inputs_mask]

        ## decide model===========================================================

        # 位置特性の正規化(線形に正規化 OR CNN)--------------------------
        if self.useSiteNormalize:
            self.siteNorm = siteNormalize()
        elif self.useSiteCNN:
            self.sCNN = []
            if self.sCNNSinglePath:
                use_site = True #TODO:Unetの各層出力にサイト特性を含ませるための変更。もう少しスマートなやり方を探したい
                self.sCNN.append(siteConv(sCNNFNum[0],7,strides=(1,1),use_bias=sCNNBias,activation=sCNNActivation))
                self.sCNN.append(siteConv(sCNNFNum[1],5,strides=(1,1),use_bias=sCNNBias,activation=sCNNActivation))
                self.sCNN.append(siteConv(sCNNFNum[3],5,strides=(1,1),use_bias=sCNNBias,activation=sCNNActivation))
                self.sCNN.append(siteConv(sCNNFNum[4],5,strides=(1,1),use_bias=sCNNBias,activation=sCNNActivation))
            else:
                # self.sCNN.append()
                self.sCNN.append(siteConv(sCNNFNum[0],5,strides=(1,1),use_bias=sCNNBias,activation=sCNNActivation))
                self.sCNN.append(siteConv(sCNNFNum[1],5,strides=(2,2),use_bias=sCNNBias,activation=sCNNActivation))
                self.sCNN.append(siteConv(sCNNFNum[2],5,strides=(2,2),use_bias=sCNNBias,activation=sCNNActivation))
                self.sCNN.append(siteConv(sCNNFNum[3],5,strides=(2,2),use_bias=sCNNBias,activation=sCNNActivation))
                self.sCNN.append(siteConv(sCNNFNum[4],5,strides=(2,2),use_bias=sCNNBias,activation=sCNNActivation))
            # チャネルが複数の場合は出力にGlobalAveragePoolingをかける
            # そうでない場合は、なにもかけない
            self.pools = [GlobalAveragePooling2D() if sCNNFNum[i] > 1 else None for i in range(len(self.sCNN))]

        #---------------------------------------------------

        # Args
        keyArgs = {"posEmbChan":posEmbChan, "use_site":use_site,"use_sCNN":useSiteCNN, "opeType":self.opeType,
         "eachChannel":eachChannel,"sConvKernelLearn":sConvKernelLearn,"site_range":site_range,
         "sklSigmoid":sklSigmoid,"sConvChan":self.sConvChan, "learnMultiSiteW":self.learnMultiSiteW}
        Args = [
            # [フィルター数,フィルターサイズ,番号]
            [encFNum[0],7, 1],
            [encFNum[1],5, 2],
            [encFNum[2],5, 3],
            [encFNum[3],3, 4],
            [encFNum[4],3, 5]
        ]
        self.PKKey = ["site_in" if (i+1 in PKConvlayer) else None for i in range(len(encFNum))]

        # model
        # エンコーダ5層
        if self.sConvChan is not None:
            keyArgs["sConvChan"] = [1,self.sConvChan]
        self.encoder1 = PKEncoder(*Args[0], **keyArgs) if (1 in PKConvlayer) else Encoder(*Args[0], bn=False)
        if self.sConvChan is not None:
            # keyArgs["sConvChan"] = [self.sConvChan]*2
            pass

        self.encoder2 = PKEncoder(*Args[1], **keyArgs) if (2 in PKConvlayer) else Encoder(*Args[1])
        self.encoder3 = PKEncoder(*Args[2], **keyArgs) if (3 in PKConvlayer) else Encoder(*Args[2])
        self.encoder4 = PKEncoder(*Args[3], **keyArgs) if (4 in PKConvlayer) else Encoder(*Args[3])
        keyArgs["sConvKernelLearn"] = False
        self.encoder5 = PKEncoder(*Args[4], **keyArgs) if (5 in PKConvlayer) else Encoder(*Args[4])
        # keyArgs["sConvKernelLearn"] = True

        # デコーダ5層
        self.decoder6 = Decoder(encFNum[3], 3)
        self.decoder7 = Decoder(encFNum[2],3)
        self.decoder8 = Decoder(encFNum[1],3)
        self.decoder9 = Decoder(encFNum[0],3)
        self.decoder10 = Decoder(3,3,bn=False)
        self.conv2d = Conv2D(1,1,activation='sigmoid',name='output_img')
        ## =======================================================================

        self.ones33 = K.ones(shape=(3, 3, 1, 1))

        # Create UNet-like model
        if self.gpus <= 1:
            self.outputs_img = self.build_pconv_unet()
            self.model = Model(inputs=self.inputs,outputs=self.outputs_img)
            self.compile_pconv_unet(self.model, self.inputs_mask,lr=self.learning_rate)
        else:
            with tf.device("/cpu:0"):
                self.model = Model(inputs=self.inputs, outputs=self.build_pconv_unet())
            self.model = multi_gpu_model(self.model, gpus=self.gpus)
            self.compile_pconv_unet(self.model, self.inputs_mask,lr=self.learning_rate)

    def build_pconv_unet(self, train_bn=True):
        self.encodeCnt = 0
        def siteExtract(outs):
            # key(site_inがあるかどうか)がNoneであるかを見て、KeyArgsを返す
            key = self.PKKey[self.encodeCnt]
            self.encodeCnt += 1
            
            # サイト特性を用いない場合
            if key==None:
                return {}
            # site CNN を使う場合（正規化CNN）
            elif self.useSiteCNN:
                if self.sCNNSinglePath: # CNNの最終出力のみ使用
                    if self.encodeCnt==self.firstLayer:
                        siteValue = self.sCNN_outs[-1]
                    else:
                        siteValue = outs[2]
                else:# CNNの途中出力を順に使用
                    # indexの開始位置： encodeCnt=1 sCNN_outs=0
                    # 第一層目は入力をかける
                    siteValue = self.sCNN_outs[self.encodeCnt-1] if  self.encodeCnt!=self.firstLayer else self.inputs[2]
            else:
                # サイト特性を使用する初レイヤーは入力から参照
                if self.encodeCnt==self.firstLayer:
                    if self.useSiteNormalize:# 正規化する場合
                        siteValue = self.siteNorm(self.inputs[2])
                    else:
                        siteValue = self.inputs[2]
                # サイト特性を前の層から受け取る場合
                else:
                    siteValue = outs[2]

            return {key:siteValue}

        # サイト特性の変換ネットワーク各層をつなぐ============
        if self.useSiteCNN:
            # pdb.set_trace()
            self.sCNN_outs = []
            sCNN_out = self.sCNN[0](self.inputs[2])

            # # Poolingによる重み付き和
            # if self.pools[0]!=None:
            #     gapAttention0 = K.expand_dims(self.pools[0](sCNN_out),axis=1)
            #     gapAttention0 = K.expand_dims(gapAttention0,axis=1)
            #     gapAttention0 = K.tile(gapAttention0,[1,self.img_rows,self.img_cols,1])
            #     sCNN_out = K.sum(sCNN_out*gapAttention0,axis=-1,keepdims=True)

            self.sCNN_outs.append(sCNN_out)

            for i in range(1, len(self.sCNN)):
                sCNN_out = self.sCNN[i](sCNN_out)
                # if self.pools[i]!=None: # Poolingによる重み付き和
                #     # devide = 2**i
                #     # site_row = int(self.img_rows/devide)
                #     # site_col = int(self.img_cols/devide)
                #     # gapAttention = K.expand_dims(self.pools[i](sCNN_out),axis=1)
                #     # gapAttention = K.expand_dims(gapAttention,axis=1)
                #     # gapAttention = K.tile(gapAttention,[1,site_row,site_col,1])
                #     # sum_sCNN_out = K.sum(sCNN_out*gapAttention,axis=-1,keepdims=True)
                #     sum_SCNN_out = K.mean(sCNN_out, axis=3, keepdims=True)
                #     self.sCNN_outs.append(sum_SCNN_out)
                # else:
                #     self.sCNN_outs.append(sCNN_out)
                self.sCNN_outs.append(sCNN_out)
            # pdb.set_trace()
        #==================================================

        # エンコーダ5層
        # siteExtractの返り値は辞書型
        e1 = self.encoder1(self.inputs[0],self.inputs[1],**siteExtract(self.inputs))
        e2 = self.encoder2(*e1[:2],**siteExtract(e1))
        e3 = self.encoder3(*e2[:2],**siteExtract(e2))
        e4 = self.encoder4(*e3[:2],**siteExtract(e3))
        e5 = self.encoder5(*e4[:2],**siteExtract(e4))

        # デコーダ5層
        d_conv6, d_mask6 = self.decoder6(e5[0], e5[1], e4[0], e4[1])
        d_conv7, d_mask7 = self.decoder7(d_conv6, d_mask6, e3[0], e3[1])
        d_conv8, d_mask8 = self.decoder8(d_conv7, d_mask7, e2[0], e2[1])
        d_conv9, d_mask9 = self.decoder9(d_conv8, d_mask8, e1[0], e1[1])
        d_conv10, _ = self.decoder10(d_conv9, d_mask9, self.inputs_img, self.inputs_mask)

        outputs = self.conv2d(d_conv10)

        return outputs

    def compile_pconv_unet(self, model, inputs_mask, lr=0.0002):
        model.compile(
            optimizer = Adam(lr=lr),
            loss= self.loss_total(inputs_mask),
            metrics=[
                self.loss_total(inputs_mask),
                self.loss_origin(inputs_mask),
                self.PSNR
            ]
        )

    def loss_total(self, mask):
        """
        Creates a loss function which sums all the loss components
        and multiplies by their weights. See paper eq. 7.
        """
        def lossFunction(y_true, y_pred):
            # Compute predicted image with non-hole pixels set to ground truth
            # Compute loss components
            origin = self.loss_origin(mask)(y_true,y_pred)

            # Return loss function
            return origin
        return lossFunction

    # partialConvolution 自体の損失関数
    def loss_origin(self,mask):
        
        def original(y_true,y_pred):

            # 観測値部分の誤差
            l1 = self.loss_valid(mask, y_true, y_pred)

            # e_mask ＝ 欠損部（陸地）：０　観測点：１　海域部：１（海洋部を損失に加えない）
            # mask = 0,1,0   exist_img = 1,1,0   1-exist_img = 0,0,1
            if self.existOff:
                if self.maskGaus is None:
                    e_mask = mask+(1-self.exist_img)
                else:
                    e_mask = mask+(1-self.expand_exist_img)
            else:
                e_mask = mask

            # 欠損部の誤差
            l2 = self.loss_hole(e_mask, y_true, y_pred)
            
            # 欠損部のまわり1pxの誤差
            y_comp = e_mask * y_true + (1-e_mask) * y_pred
            l3 = self.loss_tv(e_mask, y_comp)

            w1,w2,w3 = self.lossWeights

            res = w1*l1+w2*l2

            return tf.add(res,w3*l3,name="loss_origin")
            
        return original

    def loss_hole(self, mask, y_true, y_pred):
        """Pixel L1 loss within the hole / mask"""
        return self.l1((1-mask) * y_true, (1-mask) * y_pred)

    def loss_valid(self, mask, y_true, y_pred):
        """Pixel L1 loss outside the hole / mask"""
        return self.l1(mask * y_true, mask * y_pred)

    def loss_tv(self, mask, y_comp):
        """Total variation loss, used for smoothing the hole region, see. eq. 6"""

        # Create dilated hole region using a 3x3 kernel of all 1s.
        kernel = self.ones33
        dilated_mask = K.conv2d(1-mask, kernel, data_format='channels_last', padding='same')

        # Cast values to be [0., 1.], and compute dilated hole region of y_comp
        dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
        P = dilated_mask * y_comp

        # Calculate total variation loss
        a = self.l1(P[:,1:,:,:], P[:,:-1,:,:])# 横に1pxずらした誤差
        b = self.l1(P[:,:,1:,:], P[:,:,:-1,:])# 縦に1pxずらした誤差
        return a+b

    def fit_generator(self, generator, *args, **kwargs):
        self.model.fit_generator(
            generator,
            *args, **kwargs
        )

    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())

    def load(self, filepath, train_bn=True, lr=0.0002):
        # Load weights into model
        epoch = os.path.basename(filepath).split('.')[1].split('-')[0]
        try:
            epoch = int(epoch)
        except ValueError:
            self.current_epoch = 100
        else:
            self.current_epoch = epoch

        self.model.load_weights(filepath)

    # @staticmethod
    def PSNR(self,y_true, y_pred):
        pred = y_pred*self.exist_img
        #return 20 * K.log(4.75) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
        return - 10.0 * K.log(K.mean(K.square(pred - y_true))) / K.log(10.0)

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
 
