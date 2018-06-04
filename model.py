import numpy as np
from keras.layers import merge, Input
from keras.layers.core import Activation, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D,Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.layers import Add
from keras.models import Model,Sequential
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.engine.topology import Layer
from keras.utils import np_utils, generic_utils
from keras import backend as K
from keras.initializers import Constant
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG

def CBR(ch,shape,bn=True,sample='down',activation=LeakyReLU, dropout=False):
    model = Sequential()
    if sample=='down':
        model.add(Conv2D(filters=ch, kernel_size=(4,4), strides=2, padding='same',input_shape=shape))
    else:
        model.add(Conv2DTranspose(filters=ch, kernel_size=(4,4), strides=2, padding='same',input_shape=shape))
    if bn:
        model.add(BatchNormalization())
    if dropout:
        model.add(Dropout(0.5))
    if activation == LeakyReLU:
        model.add(LeakyReLU(alpha=0.2))
    else:
        model.add(Activation('relu'))
    return model


class FullyConvolutionalNetwork():
    def __init__(self, batchsize=1, img_height=512, img_width=512, FCN_CLASSES=5):
        self.batchsize = batchsize
        self.img_height = img_height
        self.img_width = img_width
        self.FCN_CLASSES = FCN_CLASSES
        self.vgg16 = VGG16(include_top=False,
                           weights='imagenet',
                           input_tensor=None,
                           input_shape=(self.img_height, self.img_width,3))

    def create_fcn32s(self):
        ip = Input(shape=(self.img_height, self.img_width,3))
        h = self.vgg16.layers[1](ip)
        h = self.vgg16.layers[2](h)
        h = self.vgg16.layers[3](h)
        h = self.vgg16.layers[4](h)
        h = self.vgg16.layers[5](h)
        h = self.vgg16.layers[6](h)
        h = self.vgg16.layers[7](h)
        h = self.vgg16.layers[8](h)
        h = self.vgg16.layers[9](h)
        h = self.vgg16.layers[10](h)

        h = self.vgg16.layers[11](h)
        h = self.vgg16.layers[12](h)
        h = self.vgg16.layers[13](h)
        h = self.vgg16.layers[14](h)

        h = self.vgg16.layers[15](h)
        h = self.vgg16.layers[16](h)
        h = self.vgg16.layers[17](h)
        h = self.vgg16.layers[18](h)

        h = Conv2D(self.FCN_CLASSES, (1, 1), activation='relu')(h)
        h = Conv2DTranspose(self.FCN_CLASSES,(64,64), strides=(32,32),padding="same",kernel_initializer=Constant(bilinear_upsample_weights(32,self.FCN_CLASSES)))(h)
        op = Activation('softmax')(h)
        model = Model(ip, op)
        # for layer in model.layers[:15]:
        #    layer.trainable = False
        return model

class Unet():
    def __init__(self, batchsize=1, img_height=512, img_width=512, FCN_CLASSES=5):
        self.batchsize = batchsize
        self.img_height = img_height
        self.img_width = img_width
        self.FCN_CLASSES = FCN_CLASSES

    def create_unet(self):
        ip = Input(shape=(self.img_height, self.img_width,3))

        # encoder
        h1 = Conv2D(64,   (3,3), padding= 'same', activation = 'relu')(ip)
        h = MaxPooling2D(padding='same')(h1)
        h2 = Conv2D(128,  (3,3), padding= 'same', activation = 'relu')(h)
        h = MaxPooling2D(padding='same')(h2)
        h3 = Conv2D(256,  (3,3), padding= 'same', activation = 'relu')(h)
        h = MaxPooling2D(padding='same')(h3)
        h4 = Conv2D(512,  (3,3), padding= 'same', activation = 'relu')(h)
        # h = MaxPooling2D(padding='same')(h4)
        # h5 = Conv2D(1024, (3,3), padding= 'same', activation = 'relu')(h)

        # h = Conv2DTranspose(512, (2,2), strides=2, padding='same', kernel_initializer=Constant(bilinear_upsample_weights(1,512)))(h5)
        # h = Conv2D(512,(3,3), padding= 'same', activation ='relu')(concatenate([h, h4]))
        h = Conv2DTranspose(256, (2,2), strides=2, padding='same')(h)
        h = Conv2D(256,(3,3), padding= 'same', activation ='relu')(concatenate([h, h3]))
        h = Conv2DTranspose(128, (2,2), strides=2, padding='same')(h)
        h = Conv2D(128,(3,3), padding= 'same', activation ='relu')(concatenate([h, h2]))
        h = Conv2DTranspose(64, (2,2),  strides=2, padding='same')(h)
        h = Conv2D(64,(3,3), padding= 'same', activation ='relu')(concatenate([h, h1]))

        h = Conv2D(self.FCN_CLASSES, (1, 1), activation='relu')(h)
        op = Activation('softmax')(h)
        model = Model(ip, op)
        return model

    def create_unet2(self):
        ip = Input(shape=(self.img_height, self.img_width,3))

        # encoder
        h1 = Conv2D(64,   (3,3), padding= 'same', activation = 'relu')(ip)
        h = MaxPooling2D(padding='same')(h1)
        h2 = Conv2D(128,  (3,3), padding= 'same', activation = 'relu')(h)
        h = MaxPooling2D(padding='same')(h2)
        h3 = Conv2D(256,  (3,3), padding= 'same', activation = 'relu')(h)
        h = MaxPooling2D(padding='same')(h3)
        h4 = Conv2D(512,  (3,3), padding= 'same', activation = 'relu')(h)
        h = MaxPooling2D(padding='same')(h4)
        h5 = Conv2D(1024, (3,3), padding= 'same', activation = 'relu')(h)

        h = Conv2DTranspose(512, (2,2), strides=2, padding='same')(h5)
        h = Dropout(0.5)(h)
        h = Conv2D(512,(3,3), padding= 'same', activation ='relu')(concatenate([h, h4]))
        h = Conv2DTranspose(256, (2,2), strides=2, padding='same')(h)
        h = Dropout(0.5)(h)
        h = Conv2D(256,(3,3), padding= 'same', activation ='relu')(concatenate([h, h3]))
        h = Conv2DTranspose(128, (2,2), strides=2, padding='same')(h)
        h = Dropout(0.5)(h)
        h = Conv2D(128,(3,3), padding= 'same', activation ='relu')(concatenate([h, h2]))
        h = Conv2DTranspose(64, (2,2),  strides=2, padding='same')(h)
        h = Conv2D(64,(3,3), padding= 'same', activation ='relu')(concatenate([h, h1]))

        h = Conv2D(self.FCN_CLASSES, (1, 1), activation='relu')(h)
        op = Activation('softmax')(h)
        model = Model(ip, op)
        return model

    def create_pix2pix(self):
        ip = Input(shape=(self.img_height, self.img_width, 3))
        # encoder
        h1 = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', input_shape = (512,512,3))(ip)
        h2 = CBR(128, (512, 512, 64), dropout = True)(h1)
        h3 = CBR(256, (256, 256, 128), dropout = True)(h2)
        h4 = CBR(512, (128, 128, 256), dropout = True)(h3)
        h5 = CBR(1024, (64, 64, 512), dropout = True)(h4)
        # decoder
        h = CBR(512, (32,32,1024), sample='up', activation='relu', dropout=True)(h5)
        h = CBR(256, (64,64,1024), sample='up',activation='relu',dropout=True)(concatenate([h,h4]))
        h = CBR(128, (128,128,512), sample='up',activation='relu',dropout=True)(concatenate([h,h3]))
        h = CBR(64,  (64,64,256)   , sample='up',activation='relu',dropout=True)(concatenate([h,h2]))
        h = Conv2D(filters = self.FCN_CLASSES, kernel_size=(3,3), strides=1, padding='same')(concatenate([h,h1]))
        op = Activation('softmax')(h)
        model = Model(ip, op)
        return model

    def create_pix2pix_2(self):
        ip = Input(shape=(self.img_height, self.img_width, 3))
        # encoder
        h1 = Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'same', input_shape = (512,512,3))(ip)
        h2 = CBR(128, (512, 512, 64))(h1)
        h3 = CBR(256, (256, 256, 128))(h2)
        h4 = CBR(512, (128, 128, 256))(h3)
        h5 = CBR(512, (64, 64, 512))(h4)
        h6 = CBR(512, (32, 32, 512))(h5)
        # decoder
        h = CBR(512,(16,16,512), sample='up', activation='relu', dropout=True)(h6)
        h = CBR(512, (32,32,1024), sample='up', activation='relu', dropout=True)(concatenate([h,h5]))
        h = CBR(256, (64,64,1024), sample='up',activation='relu',dropout=True)(concatenate([h,h4]))
        h = CBR(128, (128,128,512), sample='up',activation='relu',dropout=True)(concatenate([h,h3]))
        h = CBR(64,  (64,64,256)   , sample='up',activation='relu',dropout=True)(concatenate([h,h2]))
        h = Conv2D(filters = self.FCN_CLASSES, kernel_size=(3,3), strides=1, padding='same')(concatenate([h,h1]))
        op = Activation('softmax')(h)
        model = Model(ip, op)
        return model


def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = factor*2 - factor%2
    factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights
