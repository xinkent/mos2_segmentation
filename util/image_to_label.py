import os
from PIL import Image
import numpy as np
import os
import copy
# from color_map import make_color_map


label_out_path = '../data/label_binary/'
# clabel_out_path = './data/label_color/'
if not os.path.exists(label_out_path):
    os.mkdir(label_out_path)
# if not os.path.exists(clabel_out_path):
#     os.mkdir(clabel_out_path)

img_path = "/Users/shin/work/graphen/mos2_saito/color/"
nb_classes = 5
color_list = np.array([[255,0,255],[0,255,255],[255,0,0],[0,255,0],[0,0,255]]) # ピンク,水色,赤,青,緑
# color_map = make_color_map()

for name in os.listdir(img_path):
    print(name)
    img = Image.open(img_path + name)
    img = np.array(img)
    h,w,_ = img.shape
    label = np.zeros((h,w,nb_classes))
    for x in range(h):
        for y in range(w):
            pixel = img[x,y]
            dist = np.sum((color_list - pixel) ** 2, axis =1)
            cl = dist.argmin()
            label[x,y,cl] = 1
    label = label.argmax(axis=2).astype(np.uint8)
    
    # 2層 & それ以外の　学習ラベルを作る場合
    label[label !=2 ] = 0
    label[label == 2] = 1
    
    label = Image.fromarray(label, mode='P')
    palette_im = Image.open('/Users/shin/Dataset/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png')
    label.palette = copy.copy(palette_im.palette)
    label.save(label_out_path + name)

    # label_rgb= np.ones((h,w,3))
    # for i in range(nb_classes):
    #     label_rgb[label==i] = color_map[i]
    # Image.fromarray(label_rgb.astype(np.uint8)).save(clabel_out_path + name)
    # Image.fromarray(label.astype(np.uint8)).save(label_out_path+  name)
