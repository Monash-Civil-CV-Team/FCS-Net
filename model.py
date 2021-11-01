import numpy as np
import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
#import tensorflow.compat.v1 as tf
import tensorflow as tf
from keras import metrics
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add, concatenate
#from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

# dice loss defeind
def dice_loss(smooth=1.):
    def dice_loss_fixed(y_true, y_pred):
        intersection = K.sum(y_true * y_pred)
        union = K.sum(y_true)+K.sum(y_pred)
        return 1. -(2.*intersection+smooth)/(union+smooth)
    return dice_loss_fixed

def conv_block(input_tensor, filters, strides, d_rates):
    x = Conv2D(filters[0], kernel_size=1, dilation_rate=d_rates[0])(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel_size=3, strides=strides, padding='same', dilation_rate=d_rates[1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[2], kernel_size=1, dilation_rate=d_rates[2])(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters[2], kernel_size=1, strides=strides)(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)

    return x


def identity_block(input_tensor, filters, d_rates):
    x = Conv2D(filters[0], kernel_size=1, dilation_rate=d_rates[0])(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel_size=3, padding='same', dilation_rate=d_rates[1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[2], kernel_size=1, dilation_rate=d_rates[2])(x)
    x = BatchNormalization()(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)

    return x


def aspp_block(input_tensor, num_filters, rate_scale=1):
    x1 = Conv2D(num_filters, (3, 3), dilation_rate=(1 * rate_scale, 1 * rate_scale), padding="same")(input_tensor)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(num_filters, (3, 3), dilation_rate=(2 * rate_scale, 2 * rate_scale), padding="same")(input_tensor)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(num_filters, (3, 3), dilation_rate=(3 * rate_scale, 3 * rate_scale), padding="same")(input_tensor)
    x3 = BatchNormalization()(x3)

    x = Add()([x1, x2, x3])
    x = Conv2D(num_filters, (1, 1), padding="same")(x)
    return x

def FCS_Net(pretrained_weights=None, input_size=(512, 512, 3)):
    input = Input(input_size)

    x = Conv2D(64, kernel_size=3, strides=(2, 2), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, kernel_size=3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, kernel_size=3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = conv_block(x, filters=[64, 64, 256], strides=(1, 1), d_rates=[1, 1, 1])
    x = identity_block(x, filters=[64, 64, 256], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[64, 64, 256], d_rates=[1, 1, 1])

    x = conv_block(x, filters=[128, 128, 512], strides=(2, 2), d_rates=[1, 1, 1])
    x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])

    x = conv_block(x, filters=[256, 256, 1024], strides=(1, 1), d_rates=[1, 2, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])

    x = conv_block(x, filters=[512, 512, 2048], strides=(1, 1), d_rates=[1, 4, 1])
    x = identity_block(x, filters=[512, 512, 2048], d_rates=[1, 4, 1])
    x = identity_block(x, filters=[512, 512, 2048], d_rates=[1, 4, 1])

    x=aspp_block(x,num_filters=1)

    x = Conv2D(512, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(2, kernel_size=1)(x)
    x = Conv2DTranspose(2, kernel_size=(16, 16), strides=(8, 8), padding='same')(x)
    x = Activation('softmax')(x)
    model = Model(inputs=input, outputs=x)
    #model = Model(img_input, x)
    #model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
    #              loss='categorical_crossentropy',
    #              metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=5e-6), loss=[dice_loss(smooth=1.)], metrics=['accuracy'])




    model.summary()
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
        print('loaded pretrained_weights ... {}'.format(pretrained_weights))
    return model
