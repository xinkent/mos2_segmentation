import numpy as np
from keras.layers import merge, Input
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, Cropping2D,Conv2DTranspose
from keras.layers import Add
from keras.models import Model
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

    def create_fcn32s(self,binary=False):
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
        if binary:
            op = Activation('sigmoid')(h)
        else:
            op = Activation('softmax')(h)
        model = Model(ip, op)
        # for layer in model.layers[:15]:
        #    layer.trainable = False
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
