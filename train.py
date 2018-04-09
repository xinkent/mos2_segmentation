import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K
import tensorflow as tf
from PIL import Image
import os
import copy
from load_dataset import *
from model import FullyConvolutionalNetwork
import warnings
import argparse
import sys
sys.path.append('./util')
from color_map import make_color_map
# warnings.filterwarnings('ignore')

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',          '-e',  type=int,   default=100)
    parser.add_argument('--batchsize',      '-b',  type=int,   default=1)
    parser.add_argument('--train_dataset',  '-tr',             default='./data/ori/')
    parser.add_argument('--target_dataset', '-ta',             default='./data/label/')
    parser.add_argument('--lr',             '-l',  type=float, default=1e-5, )
    parser.add_argument('--out_path',       '-o')
    parser.add_argument('--binary',         '-bi', type=int,   default=0)
    parser.add_argument('--gpu', '-g', type=int, default=2)

    args = parser.parse_args()
    path_to_train    = args.train_dataset
    path_to_target   = args.target_dataset
    epoch            = args.epoch
    batchsize        = args.batchsize
    lr               = args.lr
    out              = args.out_path
    binary           = [False,True][args.binary]

    # set gpu environment
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    K.set_session(sess)

    if not os.path.exists(out):
        os.mkdir(out)
    f = open(out + '/param.txt','w')
    f.write('epoch:' + str(args.epoch) + '\n' + 'batch:' + str(batchsize) + '\n' + 'lr:' + str(lr))
    f.close()

    np.random.seed(seed=32)
    img_size = 512
    if binary:
        nb_class = 2
    else:
        nb_class = 5

    with open('./data/train.txt','r') as f:
        ls = f.readlines()
    train_names = [l.strip('\n') for l in ls]
    with open('./data/test.txt','r') as f:
        ls = f.readlines()
    test_names = [l.strip('\n') for l in ls]

    # names = os.listdir(path_to_train)
    # names = np.array([name[2:7] for name in names])
    # ind = np.random.permutation(len(names))
    # train_names = names[ind[:int(len(names) * 0.8)]]
    # test_names  = names[ind[int(len(names) * 0.8):]]

    nb_data = len(train_names)

    train_X, train_y = generate_dataset(train_names, path_to_train, path_to_target, img_size, nb_class)
    class_freq = np.array([np.sum(train_y.argmax(axis=3) == i) for i in range(nb_class)])
    class_weights = np.median(class_freq) /class_freq
    def crossentropy(y_true, y_pred):
        return K.mean(-K.sum(y_true*K.log(y_pred + 1e-7),axis=[3]),axis=[1,2])

    def weighted_crossentropy(y_true, y_pred):
        return K.mean(-K.sum((y_true*class_weights)*K.log(y_pred + 1e-7),axis=[3]),axis=[1,2])
    FCN = FullyConvolutionalNetwork(img_height=img_size, img_width=img_size,FCN_CLASSES=nb_class)
    adam = Adam(lr)
    train_model = FCN.create_fcn32s()
    # train_model = generator(nb_class)
    train_model.compile(loss=weighted_crossentropy, optimizer=adam)
    # train_model.fit_generator(generate_arrays_from_file(train_names, path_to_train, path_to_target, img_size, nb_class),
    #                                                steps_per_epoch=nb_data/1, epochs=1000)
    es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    # train_model.fit(train_X,train_y,batch_size = batchsize, epochs=epoch, validation_split=0.1, callbacks=[es_cb])
    train_model.fit(train_X,train_y,batch_size = batchsize, epochs=epoch, validation_split=0.1)
    train_model.save_weights(out + '/weights.h5')

    # test data
    # fig = plt.figure()

    # predict


    nb_test = len(test_names)
    color_map = make_color_map()
    ind = np.random.permutation(nb_test)
    mat = np.zeros([nb_class,nb_class])
    for i in ind:
        name = test_names[i]
        img   = load_data(path_to_train    + 'or' + name + '.png',  img_size, 'original')
        x     = load_data(path_to_train    + 'or' + name + '.png',  img_size, 'data')
        y     = load_data(path_to_target + 'col' +  name + '.png', img_size, 'label')
        pred = train_model.predict(x)[0].argmax(axis=2)
        y = y[0].argmax(axis=2)
        y_rgb = np.zeros((img_size,img_size,3))
        pred_rgb = np.zeros((img_size, img_size,3))
        for i in range(nb_class):
            y_rgb[y == i] = color_map[i]
            pred_rgb[pred==i] = color_map[i]

        # confusion matrix
        for r in range(img_size):
            for c in range(img_size):
                mat[y[r,c], pred[r,c]] += 1

        img.save(out + '/input_' + name + '.png')
        Image.fromarray(y_rgb.astype(np.uint8)).save(out + '/label_' + name + '.png')
        Image.fromarray(pred_rgb.astype(np.uint8)).save(out + '/pred_' + name + '.png')

    file = open(out + '/accuracy.csv','w')
    pd.DataFrame(mat).to_csv('confusion.csv')
    pixel_wise    = np.sum([mat[k,k] for k in range(nb_class)]) / np.sum(mat)
    mean_acc_list = [mat[k,k]/np.sum(mat[k,:]) for k in range(5)]
    mean_acc      = np.sum(mean_acc_list) / nb_class
    mean_iou_list = [mat[k,k] / (np.sum(mat[k,:]) + np.sum(mat[:,k]) - mat[k,k]) for k in range(5)]
    mean_iou      = np.sum(mean_iou_list) / nb_class
    file.write('pixel wize: ' + str(pixel_wise) + '\n' + 'mean acc: ' + str(mean_acc) + '\n' + 'mean iou: ' + str(mean_iou))

if __name__ == '__main__':
    train()
