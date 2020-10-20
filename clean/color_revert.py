import cv2
import numpy as np
import glob


white = np.array([255, 255, 255])
black = np.array([0, 0, 0])

def comp_dist(v, c):
    a = 0
    for i in range(3):
        a += (v[i] - c[i])**2
    return a


def run(fpath):
    img = cv2.imread(fpath, cv2.IMREAD_COLOR)
    img2 = np.zeros(img.shape)
    for i in range(len(img)):
        for j in range(len(img[i])):
            if comp_dist(img[i][j], white) < 100:
                img2[i][j] = black
            else:
                img2[i][j] = white
    cv2.imwrite(fpath.replace('out', 'out2'), img2)


for f in glob.glob('out/*.png'):
    run(f)
