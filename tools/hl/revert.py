import cv2
import numpy as np
import glob


colors = [[255, 0, 0],
          [255, 255, 0],
          [0, 0, 255],
          [0, 0, 0],
          [0, 255, 255],
          [0, 255, 0],
         ]
colors = np.array(colors)

white = np.array([255, 255, 255])
black = np.array([0, 0, 0])

def comp_dist(v, c):
    a = 0
    for i in range(3):
        a += (v[i] - c[i])**2
    return a


def run(fpath, color_check):
    img = cv2.imread(fpath, cv2.IMREAD_COLOR)
    img2 = np.zeros(img.shape)
    for i in range(len(img)):
        for j in range(len(img[i])):
            if comp_dist(img[i][j], color_check) < 1500:
                img2[i][j] = black
            else:
                img2[i][j] = white
    cv2.imwrite(fpath.replace('.jpg', 'a.jpg'), img2)

for c in range(len(colors)):
    for f in glob.glob('*_%s.jpg' % c):
        run(f, white)
