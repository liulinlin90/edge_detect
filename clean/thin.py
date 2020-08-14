import cv2 as cv
import numpy as np
import sys
import os
import copy

def filter(img, fsize=3, keep=8, strike=1):
    img_new = copy.deepcopy(img)
    x, y = img.shape
    for i in range(0, x, strike):
        for j in range(0, y, strike):
            if not (i + fsize < x) or not (j + fsize < y):
                continue
            val = []
            for tmpi in range(i, i + fsize +1):
                for tmpj in range(j, j + fsize + 1):
                    val.append(img[tmpi,tmpj])
            val.sort()
            th = val[keep-1]
            for tmpi in range(i, i + fsize +1):
                for tmpj in range(j, j + fsize + 1):
                    if img[tmpi,tmpj] > th:
                        img_new[tmpi, tmpj] = 255.0
    return img_new

def smooth(img):
    img_new = copy.deepcopy(img)
    x, y = img.shape
    for i in range(x):
        for j in range(y):
            val = []
            for tmpi in range(i - 1, i + 2):
                for tmpj in range(j - 1, j + 2):
                    if tmpi > 0 and tmpi < x and tmpj > 0 and tmpj < y:
                        val.append(img[tmpi, tmpj])
            if len(val) > 6:
                val.sort()
                val = val[:6]
            img_new[i, j] = sum(val)/len(val)
    return img_new


def rm_noise(img, weight=0.65):
    img[img > 255 * weight] = 255
    return img

fpath = sys.argv[1]
outdir = './out'
opath = os.path.join(outdir, os.path.basename(fpath))
img = cv.imread(fpath, cv.IMREAD_GRAYSCALE)
img = np.array(img, dtype=np.float32)

img = smooth(img)
img = rm_noise(img)
img = filter(img, fsize=4, keep=20, strike=2)
img = smooth(img)
cv.imwrite(opath, img)
