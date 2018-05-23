import numpy as np
import pandas as pd
import os
import copy
import argparse
import sys
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K
import tensorflow as tf
from PIL import Image
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from load_dataset import *
from model import FullyConvolutionalNetwork, Unet
sys.path.append('./util')
from color_map import make_color_map
from progressbar import ProgressBar
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
    parser.add_argument('--model',          '-m',  type=int,   default=0)
    parser.add_argument('--weight',         '-w',  type=int,   default=0)
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
    np.random.seed(seed=1)
    img_size = 512
    if binary:
        nb_class = 2
    else:
        nb_class = 5
        # nb_class = 3
    with open('./data/train.txt','r') as f:
        ls = f.readlines()
    train_names = [l.strip('\n') for l in ls]
    with open('./data/test.txt','r') as f:
        ls = f.readlines()
    test_names = [l.strip('\n') for l in ls]

    # names = os.listdir(path_to_train)
    # names = np.array([name[2:7] for name in names])
    # ind = np.random.permutation(len(names))
    # train_names = names[ind[:24]]
    # test_names  = names[ind[24:]]
    # f = open('./data/train.txt','w')
    # for name in train_names:
    #     f.write(name + '\n')
    # f.close()
    # f = open('./data/test.txt','w')
    # for name in test_names:
    #     f.write(name + '\n')
    # f.close()
    # return 0 
    

    nb_data = len(train_names)
    train_X, train_y = generate_dataset(train_names, path_to_train, path_to_target, img_size, nb_class)
    test_X,  test_y  = generate_dataset(test_names, path_to_train, path_to_target, img_size, nb_class, aug=1)

    #--------------------------------------------------------------------------------------------------------------------
    # training
    #--------------------------------------------------------------------------------------------------------------------
    class_freq = np.array([np.sum(train_y.argmax(axis=3) == i) for i in range(nb_class)])
    # class_weights = np.median(class_freq) /class_freq
    class_weights = np.mean(class_freq) /class_freq

    # FCN = FullyConvolutionalNetwork(img_height=img_size, img_width=img_size,FCN_CLASSES=nb_class)
    # unet = Unet(img_height=img_size, img_width=img_size,FCN_CLASSES=nb_class)
    # adam = Adam(lr)
    # train_model = FCN.create_fcn32s()
    # train_model = unet.create_model2()
    # train_model.compile(loss=crossentropy, optimizer=adam)
    # train_model.compile(loss=weighted_crossentropy, optimizer=adam)
    
    # es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    # train_model.fit(train_X,train_y,batch_size = batchsize, epochs=epoch, validation_split=0.1, callbacks=[es_cb])
    train_model = make_model(args.model, args.weight, img_size, nb_class,class_weights, lr) # (model(1:fcn, 2:unet, 3:unet2) ,  weight(0:no weight 1:weight)) 
    # train_model.fit(train_X,train_y,batch_size = batchsize, epochs=epoch, validation_split=0.1)
    train_model.fit(train_X,train_y,batch_size = batchsize, epochs=epoch) 
    train_model.save_weights(out + '/weights.h5')

    #--------------------------------------------------------------------------------------------------------------------
    # predict
    #--------------------------------------------------------------------------------------------------------------------
    nb_test = len(test_names)
    color_map = make_color_map()

    mat = np.zeros([nb_class,nb_class])
    pred = train_model.predict(test_X)
    pred_score = pred.reshape((-1,nb_class))[:,1]
    pred_label = pred.reshape((-1, nb_class)).argmax(axis=1)
    y          = test_y.reshape((-1, nb_class)).argmax(axis=1)
    for i in range(len(y)):
        mat[y[i],pred_label[i]] += 1
    file = open(out + '/accuracy.csv','w')
    pd.DataFrame(mat).to_csv(out + '/confusion.csv')
    pixel_wise    = np.sum([mat[k,k] for k in range(nb_class)]) / np.sum(mat)
    mean_acc_list = [mat[k,k]/np.sum(mat[k,:]) for k in range(nb_class)]
    mean_acc      = np.sum(mean_acc_list) / nb_class
    mean_iou_list = [mat[k,k] / (np.sum(mat[k,:]) + np.sum(mat[:,k]) - mat[k,k]) for k in range(nb_class)]
    mean_iou      = np.sum(mean_iou_list) / nb_class
    if binary:
        fpr, tpr, threshods = roc_curve(y, pred_score, pos_label = 1)
        auc_score = auc(fpr,tpr)
        recall       = mat[1,1] / np.sum(mat[1,:])
        precision = mat[1,1] / np.sum(mat[:,1])
        if recall == 0 and precision == 0:
            f_value = 0
        else:
            f_value    = 2 * recall * precision / (recall + precision)
        plt.plot(fpr,tpr)
        plt.savefig(out + '/ROC.png')
        file.write('pixel wize: ' + str(pixel_wise) + '\n' + 'mean acc: ' + str(mean_acc) + '\n' + 'mean iou: ' + str(mean_iou) + '\n' + 'auc: ' + str(auc_score) + '\n' + 'f_value: ' + str(f_value))
    else:
        file.write('pixel wize: ' + str(pixel_wise) + '\n' + 'mean acc: ' + str(mean_acc) + '\n' + 'mean iou: ' + str(mean_iou))
    file.close()

    #--------------------------------------------------------------------------------------------------------------------
    # visualize
    #--------------------------------------------------------------------------------------------------------------------
    for pr,y,name in zip(pred, test_y, test_names):
        pr = pr.argmax(axis=2)
        y  = y.argmax(axis=2)
        y_rgb = np.zeros((img_size,img_size,3))
        pred_rgb = np.zeros((img_size, img_size,3))
        for i in range(nb_class):
            y_rgb[y == i] = color_map[i]
            pred_rgb[pr==i] = color_map[i]
        # img.save(out + '/input_' + name + '.png')
        Image.fromarray(y_rgb.astype(np.uint8)).save(out + '/label_' + name + '.png')
        Image.fromarray(pred_rgb.astype(np.uint8)).save(out + '/pred_' + name + '.png')


def cross_valid():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',          '-e',  type=int,   default=100)
    parser.add_argument('--batchsize',      '-b',  type=int,   default=1)
    parser.add_argument('--train_dataset',  '-tr',             default='./data/ori/')
    parser.add_argument('--target_dataset', '-ta',             default='./data/label/')
    parser.add_argument('--lr',             '-l',  type=float, default=1e-5, )
    parser.add_argument('--out_path',       '-o',              default='./result/validation/')
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

    if not os.path.exists(out):
        os.mkdir(out)
    f = open(out + '/param.txt','w')

    # set gpu environment
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    K.set_session(sess)

    np.random.seed(seed=32)
    img_size = 512
    if binary:
        nb_class = 2
    else:
        nb_class = 5
        # nb_class = 3
    with open('./data/train.txt','r') as f:
        ls = f.readlines()
    train_names = np.array([l.strip('\n') for l in ls])
    with open('./data/test.txt','r') as f:
        ls = f.readlines()
    test_names = [l.strip('\n') for l in ls]
    train_names = np.concatenate([train_names, test_names])
    nb_data = len(train_names)
    random.shuffle(train_names)
    result = pd.DataFrame(np.zeros((3,1)))
    result.index = ['Unet2', 'pix2pix', 'pix2pix2']
    n_model = len(result.index)
    model_index_list = ((2,0), (3,0),(3,1))
    k_fold = KFold(n_splits = 3)
    for model_i in range(n_model):
        print(result.index[model_i])
        valid_score_list = []
        p = ProgressBar()
        for train, valid in p(k_fold.split(train_names)):
            train_X, train_y = generate_dataset(train_names[train], path_to_train, path_to_target, img_size, nb_class)
            valid_X, valid_y = generate_dataset(train_names[valid], path_to_train, path_to_target, img_size, nb_class, aug=1)
            #--------------------------------------------------------------------------------------------------------------------
            # training
            #--------------------------------------------------------------------------------------------------------------------
            class_freq = np.array([np.sum(train_y.argmax(axis=3) == i) for i in range(nb_class)])
            class_weights = np.median(class_freq) /class_freq
            if binary and class_freq[1] == 0:
                continue
            train_model = make_model(model_index_list[model_i][0],model_index_list[model_i][1],img_size, nb_class, class_weights,lr)
            train_model.fit(train_X,train_y,batch_size = batchsize, epochs=epoch,verbose=0)
            #--------------------------------------------------------------------------------------------------------------------
            # predict
            #--------------------------------------------------------------------------------------------------------------------
            mat = np.zeros([nb_class,nb_class])
            pred = train_model.predict(valid_X)
            pred_score = pred.reshape((-1,nb_class))[:,1]
            pred_label = pred.reshape((-1, nb_class)).argmax(axis=1)
            y          = valid_y.reshape((-1, nb_class)).argmax(axis=1)
            for i in range(len(y)):
                mat[y[i],pred_label[i]] += 1
            pixel_wise    = np.sum([mat[k,k] for k in range(nb_class)]) / np.sum(mat)
            mean_acc_list = [mat[k,k]/np.sum(mat[k,:]) for k in range(nb_class)]
            mean_acc      = np.sum(mean_acc_list) / nb_class
            mean_iou_list = [mat[k,k] / (np.sum(mat[k,:]) + np.sum(mat[:,k]) - mat[k,k]) for k in range(nb_class)]
            mean_iou      = np.sum(mean_iou_list) / nb_class
            if binary:
                fpr, tpr, threshods = roc_curve(y, pred_score, pos_label = 1)
                auc_score = auc(fpr,tpr)
                if (mat[1,1] == 0) or (sum(mat[:,1]) == 0):
                    f_value = 0
                else:
                    recall     = mat[1,1] / np.sum(mat[1,:])
                    precision  = mat[1,1] / np.sum(mat[:,1])
                    f_value    = 2 * recall * precision / (recall + precision)
                valid_score_list.append(f_value)
            else:
                valid_score_list.append(mean_acc)
        result.iloc[model_i, 0] = np.mean(valid_score_list)
    result.to_csv(out + str(nb_class) + 'class_' + str(epoch) + 'epoch.csv')


def make_model(i_model,i_loss, img_size, nb_class, weights, lr):
    FCN = FullyConvolutionalNetwork(img_height=img_size, img_width=img_size,FCN_CLASSES=nb_class)
    unet = Unet(img_height=img_size, img_width=img_size,FCN_CLASSES=nb_class)

    def crossentropy(y_true, y_pred):
        return K.mean(-K.sum(y_true*K.log(y_pred + 1e-7),axis=[3]),axis=[1,2])

    def weighted_crossentropy(y_true, y_pred):
        return K.mean(-K.sum((y_true*weights)*K.log(y_pred + 1e-7),axis=[3]),axis=[1,2])

    if   i_model == 0:
        model = FCN.create_fcn32s()
    elif i_model == 1:
        model = unet.create_unet()
    elif i_model == 2:
        model = unet.create_unet2()
    elif i_model == 3:
        model = unet.create_pix2pix()
    elif i_model == 4:
        model = unet.create_pix2pix_2()

    adam = Adam(lr)
    # es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    if i_loss == 0:
        model.compile(loss=crossentropy, optimizer=adam)
    if i_loss == 1:
        model.compile(loss=weighted_crossentropy, optimizer=adam)
    return model

def original_img():
    path_to_train = './data/ori/'
    out_dir = './result/experiment/original/'
    with open('./data/train.txt','r') as f:
        ls = f.readlines()
    train_names = [l.strip('\n') for l in ls]
    with open('./data/test.txt','r') as f:
        ls = f.readlines()
    test_names = [l.strip('\n') for l in ls]
    for name in test_names:
        path = path_to_train + "or{}.png".format(name)
        img = load_data(path,mode ='original')
        img.save(out_dir + "or{}.png".format(name))



if __name__ == '__main__':
    train()
    # cross_valid()
    # original_img()
