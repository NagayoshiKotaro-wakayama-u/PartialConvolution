
from keras.utils import conv_utils
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Dropout, LeakyReLU, GlobalAveragePooling2D
from keras.layers.merge import Concatenate
import keras
import pdb
import numpy as np

import tensorflow as tf

class PConv2D(Conv2D):
    def __init__(self, *args, n_channels=3, mono=False, posEmbChan=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]
        self.posEmbChan= posEmbChan

    def build(self, input_shape):        
        """Adapted from original _Conv() layer of Keras        
        param input_shape: list of dimensions for [img, mask]
        """
        
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        
        self.input_dim = input_shape[0][channel_axis]

        # Image kernel
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # Mask kernel
        self.kernel_mask = K.ones(shape=self.kernel_size + (self.input_dim, self.filters))

        # Calculate padding size to achieve zero-padding
        self.pconv_padding = (
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)), 
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)), 
        )

        # Window size - used for normalization
        self.window_size = self.kernel_size[0] * self.kernel_size[1]

        if self.use_bias:
            out_shape = self.compute_output_shape(input_shape)[0]
            bias_shape = (1, out_shape[1].value, out_shape[2].value, self.posEmbChan)
            self.bias = self.add_weight(shape=bias_shape,
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.tileNum = int(out_shape[3]/self.posEmbChan)
            # pdb.set_trace()
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        '''
        We will be using the Keras conv2d method, and essentially we have
        to do here is multiply the mask with the input X, before we apply the
        convolutions. For the mask itself, we apply convolutions with all weights
        set to 1.
        Subsequently, we clip mask values to between 0 and 1
        ''' 
        # Both image and mask must be supplied
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception('PartialConvolution2D must be called on a list of two tensors [img, mask]. Instead got: ' + str(inputs))

        # Padding done explicitly so that padding becomes part of the masked partial convolution
        images = K.spatial_2d_padding(inputs[0], self.pconv_padding, self.data_format)
        masks = K.spatial_2d_padding(inputs[1], self.pconv_padding, self.data_format)

        # Apply convolutions to mask
        mask_output = K.conv2d(
            masks, self.kernel_mask, 
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Apply convolutions to image
        img_output = K.conv2d(
            (images*masks), self.kernel, 
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Calculate the mask ratio on each pixel in the output mask
        mask_ratio = self.window_size / (mask_output + 1e-8)

        # Clip output to be between 0 and 1
        mask_output = K.clip(mask_output, 0, 1)

        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * mask_output

        # Normalize iamge output
        img_output = img_output * mask_ratio

        # Add bias
        if self.use_bias:
            # pdb.set_trace()
            # img_output = K.bias_add(
            #     img_output,
            #     self.bias,
            #     data_format=self.data_format)
            img_output = img_output + K.tile(self.bias,[1,1,1,self.tileNum])

        # Apply activations on the image
        if self.activation is not None:
            img_output = self.activation(img_output)
            
        return [img_output, mask_output]
    
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[0][1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0][0],) + tuple(new_space) + (self.filters,)
            return [new_shape, new_shape]
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0], self.filters) + tuple(new_space)
            return [new_shape, new_shape]

class Encoder(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, iterNum, bn=True, istraining=True, use_bias=False, posEmbChan=1):
        super(Encoder, self).__init__()
        self.pconv2d = PConv2D(filters, kernel_size, posEmbChan=posEmbChan, strides=2, padding='same',use_bias=use_bias)
        self.count = iterNum
        self.bn = bn
        self.training = istraining
        self.batchnorm = BatchNormalization(name='EncBN'+str(self.count))
        self.relu = Activation('relu')
        self.use_bias = use_bias

    def call(self,img_in,mask_in):
        conv,mask = self.pconv2d([img_in,mask_in])
        if self.bn:
            conv = self.batchnorm(conv,training=self.training)
        conv = self.relu(conv)
        return conv, mask

class Decoder(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, bn=True, use_bias=False):
        super(Decoder, self).__init__()
        self.bn = bn
        self.upsample = UpSampling2D(size=(2,2))
        self.concatenate = Concatenate(axis=3)
        self.pconv2d = PConv2D(filters, kernel_size, padding='same',use_bias=use_bias)
        self.batchnorm = BatchNormalization()
        self.leakyrelu = LeakyReLU(alpha=0.2)
        self.use_bias = use_bias
    
    def call(self,img_in,mask_in, e_conv, e_mask):
        up_img = self.upsample(img_in)
        up_mask = self.upsample(mask_in)
        conc_img = self.concatenate([e_conv,up_img])
        conc_mask = self.concatenate([e_mask,up_mask])
        conv,mask = self.pconv2d([conc_img,conc_mask])
        if self.bn:
            conv = self.batchnorm(conv)
        conv = self.leakyrelu(conv)
        return conv, mask

class sitePConv(PConv2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4), InputSpec(ndim=4)]

    def call(self, inputs, mask=None):

        # Both image and mask must be supplied
        if type(inputs) is not list or len(inputs) < 2 or len(inputs) > 3:
            raise Exception('site-PartialConvolution must be called on a list of two tensors [img, mask] or [img, mask,site]. Instead got: ' + str(inputs))
        
        # 学習可能なPositionEmbeddingを行うかどうかのフラグ
        if len(inputs) == 2:
            self.isLeanablePosEmb = True
            siteFilter = self.add_weight(shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        name='site_kernel',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        else:
            self.isLeanablePosEmb = False
            siteFilter = K.spatial_2d_padding(inputs[2], self.pconv_padding, self.data_format)

        # Padding done explicitly so that padding becomes part of the masked partial convolution
        images = K.spatial_2d_padding(inputs[0], self.pconv_padding, self.data_format)
        masks = K.spatial_2d_padding(inputs[1], self.pconv_padding, self.data_format)
        

        # Apply convolutions to mask
        mask_output = K.conv2d(
            masks, self.kernel_mask, # K.ones
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # サイト特性の考慮
        # siteFilter = K.tile(siteFilter,[self.input_dim,1,1,1])
        images = images + siteFilter

        # Apply convolutions to image
        img_output = K.conv2d(
            (images*masks), self.kernel,
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Calculate the mask ratio on each pixel in the output mask
        mask_ratio = self.window_size / (mask_output + 1e-8)
        # Clip output to be between 0 and 1
        mask_output = K.clip(mask_output, 0, 1)
        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * mask_output
        # Normalize iamge output
        img_output = img_output * mask_ratio


        # Apply bias only to the image (if chosen to do so)
        # pdb.set_trace()
        if self.use_bias:
            img_output = K.bias_add(
                img_output,
                self.bias,
                data_format=self.data_format)
        
        # Apply activations on the image
        if self.activation is not None:
            img_output = self.activation(img_output)
            
        return [img_output, mask_output]

class siteEncoder(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, iterNum, bn=True, istraining=True):
        super(siteEncoder, self).__init__()
        self.sitepconv = sitePConv(filters, kernel_size, strides=2, use_bias=False, padding='same')
        self.count = iterNum
        self.bn = bn
        self.training = istraining
        self.batchnorm = BatchNormalization(name='EncBN'+str(self.count))
        self.relu = Activation('relu')

    def call(self,img_in,mask_in,site_in):
        conv,mask = self.sitepconv([img_in,mask_in,site_in])
        if self.bn:
            conv = self.batchnorm(conv,training=self.training)
        conv = self.relu(conv)
        return conv, mask

class siteDecoder(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, bn=True):
        super(siteDecoder, self).__init__()
        self.bn = bn
        self.upsample = UpSampling2D(size=(2,2))
        self.concatenate = Concatenate(axis=3)
        self.sitepconv = sitePConv(filters, kernel_size, padding='same')
        self.batchnorm = BatchNormalization()
        self.leakyrelu = LeakyReLU(alpha=0.2)
    
    def call(self,img_in,mask_in,site_in, e_conv, e_mask, e_site):
        up_img = self.upsample(img_in)
        up_mask = self.upsample(mask_in)
        up_site = self.upsample(site_in)
        conc_img = self.concatenate([e_conv,up_img])
        conc_mask = self.concatenate([e_mask,up_mask])
        conc_site = self.concatenate([e_site,up_site])
        conv,mask = self.sitepconv([conc_img,conc_mask,conc_site])
        if self.bn:
            conv = self.batchnorm(conv)
        conv = self.leakyrelu(conv)
        return conv, mask


class PKConv(Conv2D):
    def __init__(self, *args, n_channels=3, mono=False, posEmbChan=1,use_site=False,use_sCNN=False,opeType="add",eachChannel=False, **kwargs):
        super().__init__(*args, **kwargs)
        if use_site or use_sCNN:
            self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4), InputSpec(ndim=4)]
        else:
            self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]
        self.posEmbChan= posEmbChan
        self.use_site = use_site
        self.use_sCNN = use_sCNN
        self.opeType = opeType
        self.eachChannel = eachChannel

    def build(self, input_shape):        
        """Adapted from original _Conv() layer of Keras        
        param input_shape: list of dimensions for [img, mask]
        """

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        
        self.input_dim = input_shape[0][channel_axis]

        # Image kernel
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
        if self.opeType == "add":
            self.onesKernel = K.constant(np.ones(kernel_shape))
        elif self.opeType == "mul":
            self.bias_initializer = keras.initializers.Ones()

        # Mask kernel
        self.kernel_mask = K.ones(shape=kernel_shape)

        # Calculate padding size to achieve zero-padding
        self.pconv_padding = (
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)), 
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)), 
        )

        # Window size - used for normalization
        self.window_size = self.kernel_size[0] * self.kernel_size[1]

        # pdb.set_trace()
        # out_shape = self.compute_output_shape(input_shape)[0]
        if self.use_site:
            self.bias = None
            self.kernel_site = K.constant(
                np.ones(self.kernel_size + (self.posEmbChan, self.posEmbChan))
                )
        elif self.use_sCNN:
            self.bias = None
        else:
            if self.eachChannel:
                bias_shape = (1, input_shape[0][1].value, input_shape[0][2].value, self.input_dim)
            else:
                bias_shape = (1, input_shape[0][1].value, input_shape[0][2].value, self.posEmbChan)
            self.bias = self.add_weight(shape=bias_shape,
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            
        self.tileNum = input_shape[0][3].value
        self.built = True
    
    def call(self, inputs, mask=None):
        
        # Both image and mask must be supplied
        if type(inputs) is not list or len(inputs) > 3 or len(inputs) <= 1:
            raise Exception('PartialConvolution2D must be called on a list of two tensors [img, mask] or [img, mask, site]. Instead got: ' + str(inputs))

        # Padding done explicitly so that padding becomes part of the masked partial convolution
        images = K.spatial_2d_padding(inputs[0], self.pconv_padding, self.data_format)
        masks = K.spatial_2d_padding(inputs[1], self.pconv_padding, self.data_format)
        if len(inputs)==3:
            bias = K.spatial_2d_padding(inputs[2], self.pconv_padding, self.data_format)
            if self.use_site:
                site_output = K.conv2d(
                    bias, self.kernel_site,
                    strides=self.strides,
                    padding='valid',
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate
                )
        else:
            bias = K.spatial_2d_padding(self.bias, self.pconv_padding, self.data_format)

        # Apply convolutions to mask
        mask_output = K.conv2d(
            masks, self.kernel_mask, 
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        if self.opeType == "add": # 位置特性を足し算(W+P)X
            # Apply convolutions to image (W*X) 
            wx = K.conv2d(
                (images*masks), self.kernel, 
                strides=self.strides,
                padding='valid',
                data_format=self.data_format,
                dilation_rate=self.dilation_rate
            )

            # 全チャネルに対して位置特性を用意する場合
            if self.eachChannel:
                # pdb.set_trace()
                px = bias*images
                px = K.conv2d(
                    px, self.onesKernel, 
                    strides=self.strides,
                    padding='valid',
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate
                )
            else:

                # 末尾に次元を追加してタイル（位置特性ごとに塊ができるようにタイルするため）
                # pdb.set_trace()
                posEmb = K.tile(tf.expand_dims(bias[:1],-1),[1,1,1,1,self.tileNum])
                posEmb = tf.reshape(posEmb,shape=[1]+images.shape[1:3]+[self.tileNum*self.posEmbChan])
                tiled_images = K.tile(images,[1,1,1,self.posEmbChan])


                # PX = P(位置特性)  ×  X(特徴マップ)
                pxs = posEmb*tiled_images

                px = 0
                for c_ind in range(self.posEmbChan):
                    pxi = pxs[:,:,:,self.tileNum*c_ind:self.tileNum*(c_ind+1)]
                    pxi = K.conv2d(
                        pxi, self.onesKernel, 
                        strides=self.strides,
                        padding='valid',
                        data_format=self.data_format,
                        dilation_rate=self.dilation_rate
                    )
                    px = pxi + px # TODO:いずれはAttenntionなどによる重みつき和を計算したい

            img_output = wx + px

        elif self.opeType == "mul": # 位置特性を掛け算
            posEmb = K.tile(bias,[1,1,1,self.tileNum])
            px = posEmb*images

            img_output = K.conv2d(
                (px*masks), self.kernel, 
                strides=self.strides,
                padding='valid',
                data_format=self.data_format,
                dilation_rate=self.dilation_rate
            )


        # Calculate the mask ratio on each pixel in the output mask
        mask_ratio = self.window_size / (mask_output + 1e-8)

        # Clip output to be between 0 and 1
        mask_output = K.clip(mask_output, 0, 1)

        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * mask_output

        # Normalize iamge output
        img_output = img_output * mask_ratio

        # Apply activations on the image
        if self.activation is not None:
            img_output = self.activation(img_output)
          
        if self.use_site:
            outputs = [img_output, mask_output, site_output]
        else:
            outputs = [img_output, mask_output]

        return outputs
    
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[0][1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0][0],) + tuple(new_space) + (self.filters,)
            return [new_shape, new_shape]
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0], self.filters) + tuple(new_space)
            return [new_shape, new_shape]

class PKEncoder(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, iterNum, posEmbChan=1, use_site=False, use_sCNN=False, opeType="add", bn=True, istraining=True,eachChannel=False):
        super().__init__()
        
        self.pkconv = PKConv(filters, kernel_size, posEmbChan=posEmbChan, use_site=use_site, use_sCNN=use_sCNN, opeType=opeType, eachChannel=eachChannel, strides=2, padding='same')
        self.count = iterNum
        self.bn = bn
        self.training = istraining
        self.batchnorm = BatchNormalization(name='EncBN'+str(self.count))
        self.relu = Activation('relu')
        self.opeType = opeType
        self.use_sCNN = use_sCNN

    def call(self,img_in,mask_in,site_in=None):
        if site_in==None:
            inputs = [img_in,mask_in]
        elif self.use_sCNN:
            inputs=[img_in,mask_in,site_in]
        else:
            inputs=[img_in,mask_in,site_in]

        output = self.pkconv(inputs)
        conv = output[0]

        if self.bn:
            conv = self.batchnorm(conv,training=self.training)
        conv = self.relu(conv)
        output[0] = conv
        
        return output


class siteNormalize(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.weight = K.variable(value=1)
        self.bias = K.variable(value=0)
        
    def call(self,site_in):
        return site_in * self.weight + self.bias

class siteConv(Conv2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = InputSpec(ndim=4)
        # Calculate padding size to achieve zero-padding
        self.pconv_padding = (
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)), 
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)), 
        )

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
    def call(self, inputs):
        # Padding done explicitly so that padding becomes part of the masked partial convolution
        # pdb.set_trace()
        images = K.spatial_2d_padding(inputs, self.pconv_padding, self.data_format)        

        # Apply convolutions to image
        img_output = K.conv2d(
            images, self.kernel,
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Add bias
        if self.use_bias:
            img_output = K.bias_add(
                img_output,
                self.bias,
                data_format=self.data_format)

        # Apply activations on the image
        if self.activation is not None:
            img_output = self.activation(img_output)
            
        return img_output

