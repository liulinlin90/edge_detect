import cv2
import numpy as np
import copy
import os
import glob

colors = [[255, 0, 0],
          [255, 255, 0],
          [0, 0, 255],
          [0, 0, 0],
          [0, 255, 255],
          [0, 255, 0],
          [255, 255, 255]
         ]
colors = np.array(colors)

def comp_dist(v, c):
    a = 0
    for i in range(3):
        a += (v[i] - c[i])**2
    return a


def get_color(v):
    dist = []
    for i in range(len(colors)):
        dist.append(comp_dist(v, colors[i]))
    val, idx = min((val, idx) for (idx, val) in enumerate(dist))
    return idx


def fill_img1(img, color):
    for i in range(len(img)):
        on = False
        pre_color = False
        for j in range(len(img[i])):
            if (img[i][j] == color).all():
                if not pre_color and on:
                    on = False
                elif not pre_color and not on:
                    on = True
                pre_color = True
            else:
                pre_color = False
            if on:
                img[i][j] = color
    return img

def fill_img1b(img, color):
    for i in range(len(img)):
        on = False
        pre_color = False
        for j in reversed(range(len(img[i]))):
            if (img[i][j] == color).all():
                if not pre_color and on:
                    on = False
                elif not pre_color and not on:
                    on = True
                pre_color = True
            else:
                pre_color = False
            if on:
                img[i][j] = color
    return img



def fill_img2(img, color):
    for j in range(len(img[0])):
        on = False
        pre_color = False
        for i in range(len(img)):
            if (img[i][j] == color).all():
                if not pre_color and on:
                    on = False
                elif not pre_color and not on:
                    on = True
                pre_color = True
            else:
                pre_color = False
            if on:
                img[i][j] = color
    return img


def fill_img2b(img, color):
    for j in range(len(img[0])):
        on = False
        pre_color = False
        for i in reversed(range(len(img))):
            if (img[i][j] == color).all():
                if not pre_color and on:
                    on = False
                elif not pre_color and not on:
                    on = True
                pre_color = True
            else:
                pre_color = False
            if on:
                img[i][j] = color
    return img


def fill_img(img, color):
    i1 = fill_img1(copy.deepcopy(img), color)
    i1b = fill_img1b(copy.deepcopy(img), color)
    i2 = fill_img2(copy.deepcopy(img), color)
    i2b = fill_img2b(copy.deepcopy(img), color)
    for i in range(len(img)):
        for j in range(len(img[i])):
            if ((i1[i][j] == color).all() and (i1b[i][j] == color).all()) \
                or ((i2[i][j] == color).all() and (i2b[i][j] == color).all()):
                img[i][j] = color
    return img

def process(fpath):
    oi = cv2.imread(fpath, cv2.IMREAD_COLOR)
    images = []
    for i in range(6):
        images.append(np.ones(oi.shape) * 255)

    for i in range(len(oi)):
        for j in range(len(oi[i])):
            c = get_color(oi[i][j])
            if c < 6:
                images[c][i][j] = colors[c]
    for i in range(6):
        images[i] = fill_img(images[i], colors[i])
    fprefix = os.path.basename(fpath).split('_')[0]
    for i in range(6):
        cv2.imwrite(fprefix + '_' + str(i) + '.jpg', images[i])


FPATH='*_off.jpg'
for f in glob.glob(FPATH):
    process(f)
