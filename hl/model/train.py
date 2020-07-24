import os
import sys
import glob
import random
import math
import re
import time
import numpy as np
import cv2


# Import Mask RCNN
from config import Config
import utils
import model as modellib
from model import log


# Directory to save logs and trained model
MODEL_DIR = '/home/linlin.liu/research/ct/data/model/hl/log'
TRAIN_DATA = ('/home/linlin.liu/research/ct/data/portrait2/train_hl/imgs/train/rgbr/real/*jpg', '/home/linlin.liu/research/ct/data/portrait2/train_hl/edge_maps/train/rgbr/real/*jpg')
VALID_DATA = ('/home/linlin.liu/research/ct/data/portrait2/train_hl/imgs/train/rgbr/real/*jpg', '/home/linlin.liu/research/ct/data/portrait2/train_hl/edge_maps/train/rgbr/real/*jpg')
TEST_DATA = ('/home/linlin.liu/research/ct/data/portrait2/train_hl/imgs/train/rgbr/real/*jpg', None)
#TEST_DATA = ('./data/test_images/*jpg', None)
SUBMISSION = './submission'

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join('/home/linlin.liu/research/ct/data/model/hl/coco', "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

def mask_to_rle(mask):
    """
    params:  mask - numpy array
    returns: run-length encoding string (pairs of start & length of encoding)
    """

    # turn a n-dimensional array into a 1-dimensional series of pixels
    # for example:
    #     [[1. 1. 0.]
    #      [0. 0. 0.]   --> [1. 1. 0. 0. 0. 0. 1. 0. 0.]
    #      [1. 0. 0.]]
    flat = mask.flatten()

    # we find consecutive sequences by overlaying the mask
    # on a version of itself that is displaced by 1 pixel
    # for that, we add some padding before slicing
    padded = np.concatenate([[0], flat, [0]])

    # this returns the indeces where the sliced arrays differ
    runs = np.where(padded[1:] != padded[:-1])[0] 
    # indexes start at 0, pixel numbers start at 1
    runs += 1

    # every uneven element represents the start of a new sequence
    # every even element is where the run comes to a stop
    # subtract the former from the latter to get the length of the run
    runs[1::2] -= runs[0::2]

    # convert the array to a string
    return ' '.join(str(x) for x in runs)


class SteelConfig(Config):
    """Configuration for training on the steel dataset.
    """
    # Give the configuration a recognizable name
    NAME = "steel"

    IMAGE_RESIZE_MODE = "crop"
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 4 

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 2

    #USE_MINI_MASK = False
    BACKBONE = "resnet50"


class SteelDataset(utils.Dataset):
    """ process steel dataset
    """

    def load_image_info(self, fpaths):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("steel", 1, "1")

        data_pattern, maskfile = fpaths
        maskinfo = {}
        if maskfile is not None:
            for f in glob.glob(data_pattern):
                img_cls = '1'
                img_id = os.path.basename(f)
                maskinfo[img_id] = {}
                maskinfo[img_id][img_cls] = f

        # Add images
        for f in glob.glob(data_pattern):
            img_id = os.path.basename(f)
            defects = []
            for img_cls in sorted(maskinfo.get(img_id, {}).keys()):
                defects.append(img_cls)
            image = cv2.imread(f, cv2.IMREAD_COLOR)
            height, width, _ = image.shape
            self.add_image("steel", image_id=img_id, path=f,
                           width=width, height=height,
                           maskinfo=maskinfo.get(img_id, {}),
                           defects=defects)


    def image_reference(self, image_id):
        """Return the defects data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "steel":
            return info["defects"]
        else:
            super(self.__class__).image_reference(self, image_id)


    def load_image(self, image_id):
        info = self.image_info[image_id]
        oriimg = cv2.imread(info['path'], cv2.IMREAD_COLOR)
        return oriimg


    def load_mask(self, image_id):
        info = self.image_info[image_id]
        num_defect = 1
        mask = np.zeros([info['height'], info['width'], num_defect], dtype=np.uint8)
        defects = []
        for i in range(num_defect):
            defects.append(str(i+1))
            maskpath = info['maskinfo'].get(str(i+1), '')
            if maskpath.strip():
                img = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
                mask[:,:,i:i+1] = np.array(img).reshape(mask.shape[0], mask.shape[1], 1)
        class_ids = np.array([self.class_names.index(d) for d in defects])
        return mask.astype(np.bool), class_ids.astype(np.int32)


def do_train_model(init_with="coco", tune_head=True):
    config = SteelConfig()
    config.display()

    # Training dataset
    dataset_train = SteelDataset()
    dataset_train.load_image_info(TRAIN_DATA)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SteelDataset()
    dataset_val.load_image_info(VALID_DATA)
    dataset_val.prepare()

    #reate model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)
    if tune_head:
        # Train the head branches
        # Passing layers="heads" freezes all layers except the head
        # layers. You can also pass a regular expression to select
        # which layers to train by name pattern.
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=15,
                    layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also 
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=5,
                augment=True,
                layers="5+")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=5,
                augment=True,
                layers="4+")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=5,
                augment=True,
                layers="all")


def do_inference():
    class InferenceConfig(SteelConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        IMAGE_RESIZE_MODE = "crop"

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # Validation dataset
    dataset_test = SteelDataset()
    dataset_test.load_image_info(TEST_DATA)
    dataset_test.prepare()

    submit_list = []
    test_count = 0
    for image_id in dataset_test.image_ids:
        test_count += 1
        if test_count % 100 == 0:
            print('number of test files processed: ', test_count)
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_test, inference_config,
                                   image_id, use_mini_mask=False)
        x, y, _ = image.shape
        if x % 2 != 0:
            image = image[:x-1]
        if y % 2 != 0:
            image = image[:,:y-1]
        print('----------', image.shape)
        result = model.detect([image], verbose=0)[0]['masks']
        tmp = result[:, :, 0].reshape(result.shape[0], result.shape[1]) * 255.0
        tmp = tmp.astype('float32')
        tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(SUBMISSION, str(image_id)+ '.jpg'), tmp)


if __name__ == '__main__':
    #do_train_model()
    do_inference()
