import cv2 as cv
import numpy as np
import sys
import os
import copy
from scipy.ndimage import gaussian_filter

def filter(img, fsize=3, remove=8, strike=1):
    img_new = copy.deepcopy(img)
    x, y = img.shape
    for i in range(0, x, strike):
        for j in range(0, y, strike):
            if not (i + fsize < x) or not (j + fsize < y):
                continue
            val = []
            for tmpi in range(i, i + fsize):
                for tmpj in range(j, j + fsize):
                    val.append(img[tmpi,tmpj])
            val.sort()
            th = val[remove]
            for tmpi in range(i, i + fsize):
                for tmpj in range(j, j + fsize):
                    if img[tmpi,tmpj] < th:
                        img_new[tmpi, tmpj] = 0
    return img_new

def smooth2(img):
    img_new = copy.deepcopy(img)
    x, y = img.shape
    for i in range(1,x):
        for j in range(1,y):
            val = []
            for tmpi in range(i - 1, i + 1):
                for tmpj in range(j - 1, j + 1):
                    if tmpi > 0 and tmpi < x and tmpj > 0 and tmpj < y:
                        val.append(img[tmpi, tmpj])
            img_new[i, j] = sum(val)/len(val)
    return img_new


def smooth(img):
    return gaussian_filter(img, sigma=1)

def rm_noise(img, weight=0.35):
    img[img < weight] = 0
    return img

fpath = sys.argv[1]
outdir = './out'
opath = os.path.join(outdir, os.path.basename(fpath))
img = cv.imread(fpath, cv.IMREAD_GRAYSCALE)
img = 1 - np.array(img, dtype=np.float32) / 255.0

img = smooth(img)
img = rm_noise(img)
img = filter(img, fsize=5, remove=4, strike=2)
img = smooth(img)
img = rm_noise(img)

cv.imwrite(opath, (1 - img) * 255.0)
