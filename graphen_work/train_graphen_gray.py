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
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import KFold
from load_dataset_graphen import *
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
        # nb_class = 5
        nb_class = 3
    # 訓練、テスト画像の番号を設定
    # 31画像を訓練に、13画像をテストに
    num_ar = np.random.permutation([str(i).zfill(3) for i in np.arange(1, 45)])
    train_names = num_ar[:31]
    test_names = num_ar[31:]

    # 訓練の時はこの辺りのコメントを入れる
    nb_data = len(train_names)
    train_X, train_y = generate_dataset(train_names, path_to_train, path_to_target, img_size, color = 3, nb_class = nb_class, aug=20)
    test_X,  test_y  = generate_dataset(test_names, path_to_train, path_to_target, img_size, color = 0, nb_class = nb_class, aug=0)
    nb_train = train_X.shape[0]
    print('train data is ' + str(nb_train))

    #--------------------------------------------------------------------------------------------------------------------
    # training
    #--------------------------------------------------------------------------------------------------------------------
    # 訓練時は以下２つのコメントを入れる
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

    es_cb = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    train_model = make_model(args.model, args.weight, img_size, nb_class,class_weights, lr) # (model(1:fcn, 2:unet, 3:unet2) ,  weight(0:no weight 1:weight))
    train_model.fit(train_X,train_y,batch_size = batchsize, epochs=epoch)
    train_model.save_weights(out + '/weights.h5')
    # train_model.load_weights(out + '/weights.h5')

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
        precision, recall, threshold = precision_recall_curve(y, pred_score)
        fpr_tpr = np.concatenate((np.array(fpr)[:,np.newaxis], np.array(tpr)[:,np.newaxis]),axis=1)
        pr = np.concatenate((np.array(precision)[:,np.newaxis], np.array(recall)[:, np.newaxis]),axis = 1)
        np.savetxt(out + '/fpr_tpr.csv', fpr_tpr, delimiter=',')
        np.savetxt(out + '/precsion_recall.csv', pr, delimiter=',')
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
    ind = np.random.permutation(nb_train)[0:10]
    imgs = train_X[ind]
    pred_train = train_model.predict(train_X[ind])
    train_y = train_y[ind]
    for j in range(10):
        img = imgs[j] * 255.0
        pr = pred_train[j]
        y = train_y[j]
        pr = pr.argmax(axis=2)
        y  = y.argmax(axis=2)
        y_rgb = np.zeros((img_size,img_size,3))
        pred_rgb = np.zeros((img_size, img_size,3))
        for i in range(nb_class):
            y_rgb[y == i] = color_map[i]
            pred_rgb[pr==i] = color_map[i]
        Image.fromarray(img.astype(np.uint8)).save(out + '/train_input_' + str(j) + '.png')
        Image.fromarray(y_rgb.astype(np.uint8)).save(out + '/train_label_' + str(j) + '.png')
        Image.fromarray(pred_rgb.astype(np.uint8)).save(out + '/train_pred_' + str(j) + '.png')

    for pr,y,name in zip(pred, test_y, test_names):
        pr = pr.argmax(axis=2)
        y  = y.argmax(axis=2)
        y_rgb = np.zeros((img_size,img_size,3))
        pred_rgb = np.zeros((img_size, img_size,3))
        for i in range(nb_class):
            y_rgb[y == i] = color_map[i]
            pred_rgb[pr==i] = color_map[i]
        # img.save(out + '/test_input_' + name + '.png')
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
    parser.add_argument('--augtype',        '-at', type=int,   default=0)
    parser.add_argument('--gpu', '-g', type=int, default=2)

    args = parser.parse_args()
    path_to_train    = args.train_dataset
    path_to_target   = args.target_dataset
    epoch            = args.epoch
    batchsize        = args.batchsize
    lr               = args.lr
    out              = args.out_path
    binary           = [False,True][args.binary]
    augtype          = args.augtype

    if not os.path.exists(out):
        os.mkdir(out)
    f = open(out + '/param.txt','w')
    f.write('epoch:' + str(args.epoch) + '\n' + 'batch:' + str(batchsize) + '\n' + 'lr:' + str(lr))
    f.close()

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
        # nb_class = 5
        nb_class = 3
    train_names = np.random.permutation([str(i).zfill(3) for i in np.arange(1, 45)])
    nb_data = len(train_names)
    random.shuffle(train_names)
    # result = pd.DataFrame(np.zeros((4,1)))
    # result.index = ['Unet2', 'pix2pix', 'unet2_weighted', 'pix2pix_weighted']   
    
    result = pd.DataFrame(np.zeros((2,2)))
    result.index = ['Unet2', 'unet2_weighted']
    result.columns = ['pixel-wise','mean-acc']
    model_index_list = ((12,0), (12,1))
    
    # result = pd.DataFrame(np.zeros((1,1)))
    # result.index = ['unet2_weighted']
    # model_index_list = [(2,1)]

    n_model = len(result.index)
    k_fold = KFold(n_splits = 3)
    for model_i in range(n_model):
        print(result.index[model_i])
        valid_score_list = []
        p = ProgressBar()
        for train, valid in p(k_fold.split(train_names)):
            print(valid)
            train_X, train_y = generate_dataset(train_names[train], path_to_train, path_to_target, img_size, color = 1, nb_class=nb_class, aug=20)
            valid_X, valid_y = generate_dataset(train_names[valid], path_to_train, path_to_target, img_size, color = 1, nb_class=nb_class, aug=0)
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
                valid_score_list.append([pixel_wise, mean_acc])
        result.iloc[model_i] = np.mean(np.array(valid_score_list), axis=0)
    result.to_csv(out + 'val_epoch{0}_lr{1}_at{2}.csv'.format(epoch, lr, augtype))


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
    elif i_model == 12:
        model = unet.create_unet2_gray()
    elif i_model == 3:
        model = unet.create_pix2pix()
    elif i_model == 13:
        model = unet.create_pix2pix_gray()
    elif i_model == 4:
        model = unet.create_pix2pix_2()

    adam = Adam(lr)
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
    # train()
    cross_valid()
    # original_img()
