import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input


def multi_label(labels, size, nb_class):
    y = np.zeros((size,size,nb_class))
    for i in range(size):
        for j in range(size):
            y[i,j,labels[i][j]] = 1
    return y

def load_data(path, size=512, mode=None):
    img = Image.open(path)
    w,h = img.size
    lw = np.random.randint(0,(w-size)/2)
    lh = np.random.randint(0,(h-size)/2)
    img = img.crop((lw,lh, lw+size, lh +size))
    if mode == "original":
        return img
    if mode == "label":
        y = np.array(img, dtype=np.int32)
        y = multi_label(y,size, 5)
        y = np.expand_dims(y, axis=0)
        return y
    if mode == "data":
        X = image.img_to_array(img)
        X = np.expand_dims(X,axis=0)
        X = preprocess_input(X)
        return X

def load_data_aug(path, size=512, mode=None,aug=3, nb_class=5):
    data_list = []
    img = Image.open(path)
    w,h = img.size
    lw = np.random.randint(0,(w-size)/2)
    lh = np.random.randint(0,(h-size)/2)
    for i in range(aug):
        img = img.crop((lw,lh, lw+size, lh +size))
        if i == 1:
            img = ImageOps.flip(img)
        elif i == 2:
            img = ImageOps.mirror(img)
        if mode == "original":
            data_list.append(img)
        if mode == "label":
            y = np.array(img, dtype=np.int32)
            y = multi_label(y,size, nb_class)
            # y = np.expand_dims(y, axis=0)
            data_list.append(y)
        if mode == "data":
            X = image.img_to_array(img)
            # X = np.expand_dims(X,axis=0)
            # X = preprocess_input(X)
            data_list.append(X)
    data = np.array(data_list)
    if mode == "data":
        data = preprocess_input(data)
    return data

def generate_arrays_from_file(names, path_to_train, path_to_target, img_size, nb_class):
    while True:
        for name in names:
            Xpath = path_to_train + "or{}.png".format(name)
            ypath = path_to_target + "col{}.png".format(name)
            X = load_data(Xpath, img_size, mode="data")
            y = load_data(ypath, img_size, mode = "label")
            yield(X,y)


def generate_dataset(names, path_to_train, path_to_target, img_size, nb_class):
    X_list = []
    y_list = []
    for name in names:
        Xpath = path_to_train + "or{}.png".format(name)
        ypath = path_to_target + "col{}.png".format(name)
        X = load_data_aug(Xpath, img_size, mode="data", nb_class=nb_class)
        y = load_data_aug(ypath, img_size, mode = "label", nb_class=nb_class)
        X_list.append(X)
        y_list.append(y)
    return np.array(X_list).reshape([-1,img_size, img_size, 3]), np.array(y_list).reshape([-1, img_size, img_size, nb_class])
