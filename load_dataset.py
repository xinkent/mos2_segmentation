import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from imgaug import augmenters as iaa
import imgaug as ia

def multi_label(labels, size, nb_class):
    y = np.zeros((size,size,nb_class))
    for i in range(size):
        for j in range(size):
            y[i,j,labels[i][j]] = 1
    return y


def load_data(name, path_to_train, path_to_target, size=512, color = 0, nb_class=5):
    img = Image.open(path_to_train + "or{}.png".format(name))
    label = Image.open(path_to_target + "col{}.png".format(name))
    w,h = img.size
    lw = (w-size)/2
    lh = (h-size)/2
    img = img.crop((lw,lh, lw+size, lh +size))
    label = label.crop((lw,lh, lw+size, lh +size))

    if color == 1:
        img = img.convert('L')
        img = np.array(img, dtype = np.float64)
        img = img / 255.0
    else:
        img = np.array(img, dtype = np.float64)[np.newaxis,:]
        img = preprocess_input(img)

    label = np.array(label, dtype=np.int32)
    label = multi_label(label,size, nb_class)
    label = label[np.newaxis,:]
    return img, label


def load_data_aug(name, path_to_train, path_to_target, size=512, color = 0 ,aug=3, nb_class=5):
    imgs = []
    labels = []
    img_ori = Image.open(path_to_train + "or{}.png".format(name))
    label_ori = Image.open(path_to_target + "col{}.png".format(name))
    w,h = img_ori.size
    cn = iaa.SomeOf((0,1),[
        iaa.ContrastNormalization((0.7,1.3)),  # 明るさ正規化
        iaa.Sequential([iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                                   iaa.WithChannels(0, iaa.Add((-10,10))),
                                   iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
                                 ]),
        iaa.Grayscale(alpha=(0,0.5))
        ])
    for i in range(aug):
        # ランダムにcrop
        lw = np.random.randint(0,(w-size))
        lh = np.random.randint(0,(h-size))
        img = img_ori.crop((lw,lh, lw+size, lh +size))
        label = label_ori.crop((lw,lh, lw+size, lh+size))


        # ランダムに反転
        j = np.random.randint(3)
        if j == 1:
            img = ImageOps.flip(img)
            label = ImageOps.flip(label)
        elif j == 2:
            img = ImageOps.mirror(img)
            label = ImageOps.mirror(label)

        if color == 0:
            # そのまま
            img = np.array(img, dtype=np.uint8)
        if color == 1:
            # gray scale
            img = img.convert('L')
            img = np.array(img, dtype=np.uint8)
        if color == 2:
            # augmentation
            img = np.array(img, dtype=np.uint8)
            img = cn.augment_image(img)

        # ランダムにratate
        t = np.random.rand() * 90
        rotate = iaa.Affine(rotate=(t,t))
        img = rotate.augment_image(img)
        label = np.array(label, dtype=np.int32)
        label = rotate.augment_image(label)

        label = multi_label(label,size, nb_class)
        imgs.append(img)
        labels.append(label)

    labels = np.array(labels)
    imgs = np.array(imgs, dtype=np.float64)
    if color == 1:
        imgs = imgs / 255.0
    else:
        imgs = preprocess_input(imgs)
    return imgs, labels

def generate_dataset(names, path_to_train, path_to_target, img_size, color, nb_class, aug=3):
    X_list = []
    y_list = []
    if color == 1:
        channel = 1
    else:
        channel = 3
    for name in names:
        if aug == 0:
            X,y =load_data(name, path_to_train, path_to_target, img_size, color, nb_class=nb_class)
        else:
            X, y = load_data_aug(name, path_to_train, path_to_target, img_size, color, nb_class=nb_class, aug=aug)
        X_list.append(X)
        y_list.append(y)
    return np.array(X_list).reshape([-1,img_size, img_size, channel]), np.array(y_list).reshape([-1, img_size, img_size, nb_class])
