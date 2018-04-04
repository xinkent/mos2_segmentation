import numpy as np
import os
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras import backend as K
from PIL import Image
import os
import copy
from load_dataset import *
from model import FullyConvolutionalNetwork
import warnings
import argparse
# warnings.filterwarnings('ignore')

def make_color_map():
    n = 256
    cmap = np.zeros((n, 3)).astype(np.int32)
    for i in range(0, n):
        d = i - 1
        r,g,b = 0,0,0
        for j in range(0, 7):
            r = bitor(r, shift_bit(get_bit(d, 0), 7 - j))
            g = bitor(g, shift_bit(get_bit(d, 1), 7 - j))
            b = bitor(b, shift_bit(get_bit(d, 2), 7 - j))
            d = shift_bit(d, -3)
        cmap[i, 0] = b
        cmap[i, 1] = g
        cmap[i, 2] = r
    return cmap[1:22]


np.random.seed(seed=32)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',          '-e',  type=int,   default=100)
    parser.add_argument('--batchsize',      '-b',  type=int,   default=1)
    parser.add_argument('--train_dataset',   '-tr',             default='./data/ori/')
    parser.add_argument('--target_dataset', '-ta',             default='./data/label/')
    parser.add_argument('--lr',             '-l',  type=float, default=1e-5, )
    parser.add_argument('--out_path',       '-o')

    args = parser.parse_args()
    path_to_train    = args.train_dataset
    path_to_target   = args.target_dataset
    epoch            = args.epoch
    batchsize        = args.batchsize
    lr               = args.lr
    out              = args.out_path

    if not os.path.exists(out):
        os.mkdir(out)
    f = open('param.txt','w')
    f.write('epoch:' + str(args.epoch) + '\n' + 'batch:' + str(batchsize) + '\n' + 'lr:' + str(lr))
    f.close()

    img_size = 512
    nb_classes = 5
    names = os.listdir(path_to_train)
    names = np.array([name[2:7] for name in names])
    ind = np.random.permutation(len(names))
    train_names = names[ind[:int(len(names) * 0.8)]]
    test_names  = names[ind[int(len(names) * 0.8):]]
    nb_data = len(train_names)

    def crossentropy(y_true, y_pred):
        return K.mean(-K.sum(y_true*K.log(y_pred + 1e-7),axis=[3]),axis=[1,2])

    train_X, train_y = generate_dataset(train_names, path_to_train, path_to_target, img_size, nb_classes)

    FCN = FullyConvolutionalNetwork(img_height=img_size, img_width=img_size,FCN_CLASSES=nb_classes)
    adam = Adam(lr)
    train_model = FCN.create_fcn32s()
    # train_model = generator(nb_classes)
    train_model.compile(loss=crossentropy, optimizer=adam)
    # train_model.fit_generator(generate_arrays_from_file(train_names, path_to_train, path_to_target, img_size, nb_classes),
    #                                                steps_per_epoch=nb_data/1, epochs=1000)
    train_model.fit(train_X,train_y,batch_size = batchsize, epochs=epoch)
    train_model.save_weights(out + '/weights.h5')


    # test data
    # fig = plt.figure()

    # predict


    nb_test = len(test_names)
    color_map = make_color_map()
    ind = np.random.permutation(nb_test)
    for i in ind:
        name = test_names[i]
        img   = load_data(path_to_train    + 'or' + name + '.png',  img_size, 'original')
        x     = load_data(path_to_train    + 'or' + name + '.png',  img_size, 'data')
        y     = load_data(path_to_target + 'col' +  name + '.png', img_size, 'label')
        pred = train_model.predict(x)[0].argmax(axis=2)
        y = y[0].argmax(axis=2)
        # pred = Image.fromarray(pred, mode='P')
        # y    = Image.fromarray(y, mode = 'P')
        # palette_im = Image.open('./data/label/col00000.png')
        # pred.palette = copy.copy(palette_im.palette)
        # y.palette    = copy.copy(palette_im.palette)
        # pred.save('./data/result/pred_' + name + '.png')
        # y.save('./data/result/label_' + name + '.png')
        y_rgb = np.zeros((img_size,img_size,3))
        pred_rgb = np.zeros((img_size, img_size,3))
        for i in range(nb_classes):
            y_rgb[y == i] = color_map[i]
            pred_rgb[pred==i] = color_map[i]
        img.save(out + '/input_' + name + '.png')
        Image.fromarray(y_rgb.astype(np.uint8)).save(out + '/label_' + name + '.png')
        Image.fromarray(pred_rgb.astype(np.uint8)).save(out + '/pred_' + name + '.png')

if __name__ == '__main__':
    train()
