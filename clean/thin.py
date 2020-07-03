import cv2 as cv
import numpy as np
import sys
import os

def filter(img, fsize=3, keep=8):
    x, y = img.shape
    for i in range(x):
        for j in range(y):
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
                        img[tmpi, tmpj] = 255.0
    return img




fpath = sys.argv[1]
outdir = './out'
opath = os.path.join(outdir, os.path.basename(fpath))
img = cv.imread(fpath, cv.IMREAD_GRAYSCALE)
img = np.array(img, dtype=np.float32)

img = filter(img, fsize=5, keep=16)
img = filter(img, fsize=3, keep=8)
cv.imwrite(opath, img)
