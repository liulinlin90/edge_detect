import cv2
import glob

def process(fpath):
    oi = cv2.imread(fpath, cv2.IMREAD_COLOR)
    cv2.imwrite(fpath.replace('.png', '_crop.png'), oi[:1024,:1024])

FPATH='*png'
for f in glob.glob(FPATH):
    process(f)
