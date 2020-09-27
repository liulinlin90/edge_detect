import os
import cv2
import glob
import random
import numpy as np
from scipy import ndimage

def rotate(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def process(fpath, shadows):
    oi = cv2.imread(fpath, cv2.IMREAD_COLOR)
    shadow = shadows[random.randint(0, len(shadows)) -1]
    shadow = ndimage.rotate(shadow, random.randint(-90,90))
    w,h,c = oi.shape
    shadow = cv2.resize(shadow, (h, w))
    sratio = random.randint(5,80)/ 100.0
    mask = load_mask(fpath)
    mask = cv2.resize(mask, (h,w))
    for i in range(w):
        for j in range(h):
            if (shadow[i,j] != [0, 0, 0]).any() and (mask[i,j] != [0, 0, 0]).any():
                oi[i,j] = oi[i,j] * sratio
    cv2.imwrite(fpath.replace('.jpg', '_da.jpg'), oi)


def load_mask(fname):
    fname = os.path.basename(fname)
    fno = fname.split('.')[0]
    f = './mask/' + fno.zfill(5) + '_skin.png'
    return cv2.imread(f, cv2.IMREAD_COLOR)


def load_shadows(fpath='./ref/*png'):
    shadows = []
    for f in glob.glob(fpath):
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        shadows.append(img)
    return shadows


shadows = load_shadows()
FPATH='free/*jpg'
for f in glob.glob(FPATH):
    if not f.endswith('_da.jpg') and not os.path.isfile(f.replace('.jpg', '_da.jpg')):
        print('processing ', f)
        process(f, shadows)

