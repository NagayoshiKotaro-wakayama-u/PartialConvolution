import os
import sys
import numpy as np
from datetime import datetime
import pdb

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, Lambda, Multiply, Dense, Flatten
from keras.layers.merge import Concatenate
from keras import backend as K
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback,Callback
from keras_tqdm import TQDMCallback

from libs.pconv_layer import PConv2D,Encoder,Decoder,sitePConv,siteEncoder,siteDecoder,PKEncoder
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

    def __init__(self, img_rows=512, img_cols=512, inference_only=False, net_name='default', gpus=1, thre=0.2, KLthre=0.1, histKLthre=0.05,
     isUsedKL=True, isUsedHistKL=True, isUsedLLH=True,LLHonly=False,exist_point_file="", exist_flag=False,
    histFSize=64,histSSize=4, truefirst= False, predfirst=False,  KLbias=True, KLonly=False):
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
        self.thre = thre
        self.KLthre = KLthre
        self.histKLthre = histKLthre

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

        self.existFlag = exist_flag
        self.histFSize = histFSize
        self.histSSize = histSSize

        # X座標,Y座標の行列
        self.Xmap = K.constant(np.tile([[i for i in range(self.img_cols)]],(self.img_rows,1))[np.newaxis,:,:,np.newaxis])
        self.Ymap = K.constant(np.tile(np.array([[i for i in range(self.img_rows)]]).T, (1,self.img_cols))[np.newaxis,:,:,np.newaxis])

        # 存在する点が１・その他が０である入力と同サイズの画像を設定
        if exist_point_file=="":
            exist_img = np.ones([self.img_rows,self.img_cols,1])
        else:
            exist_img = np.array(Image.open(exist_point_file))[np.newaxis,:,:,np.newaxis]/255
        
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
            self.compile_pconv_unet(self.model, self.inputs_mask)
            # self.loss = loss_total(self.inputs_mask)(self.)
        else:
            with tf.device("/cpu:0"):
                self.model = Model(inputs=[self.inputs_img, self.inputs_mask], outputs=self.build_pconv_unet())
            self.model = multi_gpu_model(self.model, gpus=self.gpus)
            self.compile_pconv_unet(self.model, self.inputs_mask)

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
                self.loss_KL,
                self.PSNR,
                self.loss_Djs
                # self.loss_likelihood()
            ]
            # metrics=[self.PSNR,self.loss_KL,self.loss_spatialHistKL,self.loss_origin(inputs_mask)]
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
            l4 = self.loss_KL(y_true, y_pred)
            l5 = self.loss_spatialHistKL(y_true,y_pred)
            l6 = self.loss_likelihood()(y_true,y_pred)

            if self.isUsedKL:
                if self.KLonly:
                    rslt = tf.identity(l4,name='loss_total')
                else:
                    rslt = tf.add(origin,l4,name="loss_total")
            elif self.isUsedHistKL:
                rslt = tf.add(origin,l5,name="loss_total")
            elif self.isUsedLLH:
                if self.LLHonly:
                    rslt = tf.identity(l6,name="loss_total")
                else:
                    rslt = tf.add(origin,l6,name="loss_total")
            else:
                rslt = tf.identity(origin,name="loss_total")
            
            # Return loss function
            return rslt
        return lossFunction

    # partialConvolution 自体の損失関数
    def loss_origin(self,mask):
        def original(y_true,y_pred):

            l1 = self.loss_valid(mask, y_true, y_pred)
            l2 = self.loss_hole(mask, y_true, y_pred)
            
            y_comp = mask * y_true + (1-mask) * y_pred
            l3 = self.loss_tv(mask, y_comp)

            res = l1+6*l2

            return tf.add(res,0.1*l3,name="loss_origin")
            
        return original

    def loss_likelihood(self,smallV=1e-10):
        def loglikelihood(y_true,y_pred):
            # pdb.set_trace()
            y_pred = y_pred*self.exist
            # p1 = tf.nn.relu(y_true - self.KLthre)
            p2 = tf.nn.relu(y_pred - self.KLthre)
            likelihood = -tf.log(y_true*p2)
            return tf.reduce_sum(likelihood+smallV)
        return loglikelihood

    # assume gaussian
    def loss_KL(self,y_true, y_pred,dim=2):
        def calc_norm(x):
            # pdb.set_trace()
            bias = x/(tf.reduce_sum(x,axis=(1,2,3),keepdims=True)+1e-10)
            meanX = tf.reduce_sum(bias*self.Xmap,axis=(1,2,3),keepdims=True)
            meanY = tf.reduce_sum(bias*self.Ymap,axis=(1,2,3),keepdims=True)
            disX = tf.abs((self.Xmap-meanX)*self.exist)
            disY = tf.abs((self.Ymap-meanY)*self.exist)
            sigma11 = tf.reduce_sum((disX**2) * bias,axis=(1,2,3),keepdims=True)
            sigma12 = tf.reduce_sum((disX*disY) * bias,axis=(1,2,3),keepdims=True)
            sigma22 = tf.reduce_sum((disY**2) * bias,axis=(1,2,3),keepdims=True)
            
            return meanX,meanY,sigma11,sigma12,sigma22

        # y_predの中でthre以上の値の座標を取り出すための
        # softな閾値処理
        pred = y_pred*self.exist # shape=[N,512,512,1]

        # y_trueの中でthre以上の値の座標を取り出すためのマスクを作成
        true = y_true*self.exist

        muX1,muY1,sigma11_1,sigma12_1,sigma22_1 = calc_norm(pred)
        muX2,muY2,sigma11_2,sigma12_2,sigma22_2 = calc_norm(true)

        # 分散共分散行列
        cov1 = [[ sigma11_1, sigma12_1],
                [ sigma12_1, sigma22_1]]

        cov2 = [[ sigma11_2, sigma12_2],
                [ sigma12_2, sigma22_2]]

        # 多変量正規分布(2変量)のKL-Divergenceを計算
        # 第一項 : shape=[N]
        det1 = calcDet(cov1,dim)
        det2 = calcDet(cov2,dim)

        # 第二項 : shape=[N]
        tr21 = (cov1[0][0]*cov2[1][1] - 2*cov1[0][1]*cov2[0][1] + cov1[1][1]*cov2[0][0])/(det2+1e-10)

        # 第三項 : shape=[N]
        d_mu = [tf.squeeze(muX1-muX2,axis=[1,2,3]), tf.squeeze(muY1-muY2,axis=[1,2,3])]
        sq = ((d_mu[0]**2)*cov2[1][1] - 2*d_mu[0]*d_mu[1]*cov2[0][1] + (d_mu[1]**2)*cov2[0][0] )/(det2+1e-10)

        KL = 0.5*(tf.math.log(det2+1e-10)-tf.math.log(det1+1e-10) + tr21 + sq -dim)
        return KL

    # 空間的ヒストグラム(重み付き)のKL距離
    def loss_spatialHistKL(self,y_true,y_pred):
        # pdb.set_trace()
        p_true = self.compSpatialHist(y_true, kSize=self.histFSize, thre=self.histKLthre)
        p_pred = self.compSpatialHist(y_pred, kSize=self.histFSize, thre=self.histKLthre)
        return self.compKL(p_true,p_pred)

    # KL距離
    def loss_KL_old(self,y_true,y_pred):
        return self.compKL(y_true,y_pred)

    def compKL(self,p1,p2,smallV=1e-10): # p1:true , p2:pred
        shape = tf.shape(p1) # shape=[N,w*h*c]

        # 正規化(和が１になるように)
        p1_reshape = tf.reshape(p1,[shape[0],-1])/(tf.reduce_sum(p1)+smallV)
        p2_reshape = tf.reshape(p2,[shape[0],-1])/(tf.reduce_sum(p2)+smallV)

        if self.truefirst: # 勾配が -t/p
            kl = p1_reshape*(tf.math.log(p1_reshape+smallV) - tf.math.log(p2_reshape+smallV))
            if self.KLbias:
                kl = kl + p2_reshape # 勾配が 1-t/p になる
        elif self.predfirst: # 勾配が 1+log(p/t)
            kl = p2_reshape*(tf.math.log(p2_reshape+smallV) - tf.math.log(p1_reshape+smallV))
            if self.KLbias:
                kl = kl - p2_reshape # 勾配が log(p/t) になる
        else:
            # Jensen Shanon Divergence : KL(t|p)+KL(p|t)
            kl_tp = p1_reshape*(tf.math.log(p1_reshape+smallV) - tf.math.log(p2_reshape+smallV))
            kl_pt = p2_reshape*(tf.math.log(p2_reshape+smallV) - tf.math.log(p1_reshape+smallV))

            kl = kl_tp + kl_pt

        kl = tf.reduce_sum(kl,axis=1)
        return kl


    def loss_Djs(self,y_true,y_pred):
        smallV = 1e-10
        shape = tf.shape(y_true) # shape=[N,w*h*c]

        # 正規化(和が１になるように)
        p1_reshape = tf.reshape(y_true,[shape[0],-1])/(tf.reduce_sum(y_true)+smallV)
        p2_reshape = tf.reshape(y_pred,[shape[0],-1])/(tf.reduce_sum(y_pred)+smallV)
        # Jensen Shanon Divergence : KL(t|p)+KL(p|t)
        kl_tp = p1_reshape*(tf.math.log(p1_reshape+smallV) - tf.math.log(p2_reshape+smallV))
        kl_pt = p2_reshape*(tf.math.log(p2_reshape+smallV) - tf.math.log(p1_reshape+smallV))
        Djs = kl_tp + kl_pt

        return tf.reduce_sum(Djs,axis=1)

    def compSpatialHist(self,x, kSize=64, sSize=4, isNormMode='sum', thre=0.0):
        # binarize images
        # x_bin = x*self.exist
        x = tf.nn.relu(x - thre)

        # kernel with all ones
        kernel = np.ones([kSize,kSize,1,1])
        kernel = tf.constant(kernel, dtype=tf.float32)
        
        # histogram using conv2d
        x_conv = tf.nn.conv2d(x,kernel,strides=[1,sSize,sSize,1],padding='VALID')
        shape = tf.shape(x_conv)
        x_conv_flat = tf.reshape(x_conv,[shape[0],shape[1]*shape[2]])

        if isNormMode == 'max':
            x_conv_flat = x_conv_flat/tf.reduce_max(x_conv_flat,axis=1,keepdims=True)
        elif isNormMode == 'sum':
            x_conv_flat = x_conv_flat/tf.reduce_sum(x_conv_flat,axis=1,keepdims=True)

        x_conv = tf.reshape(x_conv_flat,[shape[0],shape[1],shape[2]])
        return x_conv

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


class sitePConvUnet(object):

    def __init__(self, img_rows=512, img_cols=512, use_site=False, inference_only=False, net_name='default', gpus=1,
     exist_point_file="", exist_flag=False, posEmbChan=1):
        # Settings
        self.img_rows = img_rows
        self.img_cols = img_cols
        #self.img_overlap = 30
        self.inference_only = inference_only
        self.net_name = net_name
        self.gpus = gpus
        self.losses = None
        self.use_site = use_site

        self.existFlag = exist_flag

        # 存在する点が１・その他が０である入力と同サイズの画像を設定
        if exist_point_file=="":
            exist_img = np.ones([self.img_rows,self.img_cols,1])
        else:
            exist_img = np.array(Image.open(exist_point_file))[np.newaxis,:,:,np.newaxis]/255
        
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
            self.compile_pconv_unet(self.model, self.inputs_mask)
        else:
            with tf.device("/cpu:0"):
                self.model = Model(inputs=inputs, outputs=self.build_pconv_unet())
            self.model = multi_gpu_model(self.model, gpus=self.gpus)
            self.compile_pconv_unet(self.model, self.inputs_mask)

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

            l1 = self.loss_valid(mask, y_true, y_pred)
            l2 = self.loss_hole(mask, y_true, y_pred)
            
            y_comp = mask * y_true + (1-mask) * y_pred
            l3 = self.loss_tv(mask, y_comp)

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
 

class PKConvUnet(object):

    def __init__(self, img_rows=512, img_cols=512, use_site=True, inference_only=False, net_name='default', gpus=1,
     exist_point_file="", exist_flag=False, posEmbChan=1,opeType="add",PKConvlayer=[3,4,5],encFNum=[64,128,256,512,512],eachChannel=False):
        # Settings
        self.img_rows = img_rows
        self.img_cols = img_cols
        #self.img_overlap = 30
        self.inference_only = inference_only
        self.net_name = net_name
        self.gpus = gpus
        self.losses = None
        self.use_site = use_site # 既存のサイト特性を使用するか否か
        self.opeType = opeType
        self.posEmbChan = posEmbChan
        self.firstLayer = PKConvlayer[0]
        self.existFlag = exist_flag

        # 存在する点が１・その他が０である入力と同サイズの画像を設定
        if exist_point_file=="":
            exist_img = np.ones([self.img_rows,self.img_cols,1])
        else:
            exist_img = np.array(Image.open(exist_point_file))[np.newaxis,:,:,np.newaxis]/255
        
        self.exist = K.constant(exist_img)
        self.obsNum = np.sum(exist_img)

        # Set current epoch
        self.current_epoch = 0

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # INPUTS
        self.inputs_img = Input((self.img_rows, self.img_cols, 1), name='inputs_img')
        self.inputs_mask = Input((self.img_rows, self.img_cols, 1), name='inputs_mask')


        if use_site:# 位置特性を初めの入力に用いているかどうか
            # pdb.set_trace()
            devide = 2**(self.firstLayer-1)
            site_row = int(self.img_rows/devide)
            site_col = int(self.img_cols/devide)
            self.inputs_site = Input((site_row, site_col, posEmbChan), name='inputs_site')
            self.inputs = [self.inputs_img, self.inputs_mask,self.inputs_site]
        else:
            self.inputs = [self.inputs_img, self.inputs_mask]

        ## decide model===========================================================
        # Args
        keyArgs = {"posEmbChan":posEmbChan, "use_site":use_site, "opeType":self.opeType, "eachChannel":eachChannel}
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
        self.encoder1 = PKEncoder(*Args[0], **keyArgs) if (1 in PKConvlayer) else Encoder(*Args[0], bn=False)
        self.encoder2 = PKEncoder(*Args[1], **keyArgs) if (2 in PKConvlayer) else Encoder(*Args[1])
        self.encoder3 = PKEncoder(*Args[2], **keyArgs) if (3 in PKConvlayer) else Encoder(*Args[2])
        self.encoder4 = PKEncoder(*Args[3], **keyArgs) if (4 in PKConvlayer) else Encoder(*Args[3])
        self.encoder5 = PKEncoder(*Args[4], **keyArgs) if (5 in PKConvlayer) else Encoder(*Args[4])

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
            self.compile_pconv_unet(self.model, self.inputs_mask)
        else:
            with tf.device("/cpu:0"):
                self.model = Model(inputs=self.inputs, outputs=self.build_pconv_unet())
            self.model = multi_gpu_model(self.model, gpus=self.gpus)
            self.compile_pconv_unet(self.model, self.inputs_mask)

    def build_pconv_unet(self, train_bn=True):
        self.encodeCnt = 0
        def siteExtract(outs):
            # key(site_inがあるかどうか)がNoneであるかを見て、KeyArgsを返す
            key = self.PKKey[self.encodeCnt]
            self.encodeCnt += 1
            # pdb.set_trace()
            if key==None:
                return {}
            elif self.encodeCnt==self.firstLayer and self.use_site:
                # サイト特性を使用する初レイヤーは入力から参照
                siteValue = self.inputs[2]
            else:
                siteValue = outs[2] if len(outs)==3 else None
            
            return {key:siteValue}

        # pdb.set_trace()
        e1 = self.encoder1(self.inputs[0],self.inputs[1],**siteExtract(self.inputs))
        e2 = self.encoder2(*e1[:2],**siteExtract(e1))
        e3 = self.encoder3(*e2[:2],**siteExtract(e2))
        e4 = self.encoder4(*e3[:2],**siteExtract(e3))
        e5 = self.encoder5(*e4[:2],**siteExtract(e4))

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

            l1 = self.loss_valid(mask, y_true, y_pred)
            l2 = self.loss_hole(mask, y_true, y_pred)
            
            y_comp = mask * y_true + (1-mask) * y_pred
            l3 = self.loss_tv(mask, y_comp)

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
        self.model.fit_generator(
            generator,
            *args, **kwargs
        )

    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())

    def load(self, filepath, train_bn=True, lr=0.0002):
        # Load weights into model
        epoch = int(os.path.basename(filepath).split('.')[1].split('-')[0])
        assert epoch > 0, "Could not parse weight file. Should include the epoch"
        self.current_epoch = epoch
        self.model.load_weights(filepath)

    @staticmethod
    def PSNR(y_true, y_pred):
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
 
