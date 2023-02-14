import cv2
import tensorflow as tf
import argparse
import importlib
import os
import numpy as np
import pandas as pd
from models.condGAN256 import condGAN256
from models.cycleGAN256 import cycleGAN256

def createDir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def getGeneral(mask1, mask2):
    intersectionA = cv2.bitwise_and(mask1,mask2)
    unionA = cv2.bitwise_or(mask1,mask2)

    mask1 = cv2.bitwise_not(mask1)
    mask2 = cv2.bitwise_not(mask2)

    intersectionB = cv2.bitwise_and(mask1,mask2)
    unionB = cv2.bitwise_or(mask1,mask2)

    return intersectionA, unionA, intersectionB, unionB

def getDice(general, size):
    intersectionA, unionA, intersectionB, unionB = general
    diceA = 2*np.sum(intersectionA)/(2*size*255)
    diceB = 2*np.sum(intersectionB)/(2*size*255)
    return (diceA+diceB)/2

def getDice2(general, size):
    intersectionA, unionA, intersectionB, unionB = general
    dice = 2*np.sum(intersectionA)/(np.sum(unionA)+np.sum(intersectionA))
    return dice

def getIoU(general):
    intersectionA, unionA, intersectionB, unionB = general
    iouA = np.sum(intersectionA)/np.sum(unionA)
    iouB = np.sum(intersectionB)/np.sum(unionB)
    return (iouA+iouB)/2

def getIoUA(general):
    intersectionA, unionA, intersectionB, unionB = general
    iouA = np.sum(intersectionA)/np.sum(unionA)
    return iouA

def getIoUB(general):
    intersectionA, unionA, intersectionB, unionB = general
    iouB = np.sum(intersectionB)/np.sum(unionB)
    return iouB

def getPixelAccuracy(general, size):
    intersectionA, _, intersectionB, _ = general
    return (np.sum(intersectionA)+np.sum(intersectionB))/(size*255)

def segmentation_metrics(mask1,mask2):
    general = getGeneral(mask1,mask2)
    acc = getPixelAccuracy(general, mask1.size)
    iou = getIoU(general)
    iouA = getIoUA(general)
    iouB = getIoUB(general)

    dice = getDice(general,mask1.size)
    dice2 = getDice2(general,mask1.size)

    return acc, iou, iouA, iouB, dice, dice2

def load_triple(image_file):

  image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
  _, w = np.shape(image)
  w = w // 3
  image = np.array(image>128 + 0, dtype='uint8')

  input_image = image[:, :w]
  gt = image[:, w:(2*w)]
  prediction = image[:, (2*w):]

  gt = gt * 255
  prediction = prediction * 255

  return input_image, gt, prediction

def load_image_segmentation(image_file):
  input_image, gt, prediction  = load_triple(image_file)

  return input_image, gt, prediction

def MT2F_CLR(input_image):
    #im = np.array(input_image*255, dtype='uint8')
    im = input_image
    roi_hue = [170, 260]
    roi_hue_255 = [int(round(k*255/360,0)) for k in roi_hue]
    #roi_hue_255 = [110, 130]
    roi_hue_255 = [80, 150]

    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    low_blue = (roi_hue_255[0],50,50)
    high_blue = (roi_hue_255[1],255,255)

    blue_mask = cv2.inRange(im_hsv, low_blue, high_blue)
    blue_gt = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
    #cv2.imshow('fibrosis', cv2.hconcat((im, blue_gt)))
    #cv2.waitKey()

    return blue_gt

class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5, **kwargs):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def get_config(self):
    config = super().get_config().copy()
    return config

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset

#cGAN = tf.keras.models.load_model("F:/N002-Research/liver-pathology/segmentation/transformation_models/cGAN.h5")
#cycleGAN = tf.keras.models.load_model("F:/N002-Research/liver-pathology/segmentation/transformation_models/cycleGAN.h5", custom_objects={'InstanceNormalization': InstanceNormalization})

checkpoint_path = "F:/N002-Research/liver-pathology/segmentation/transformation_models/cycleGAN/ckpt-7"
cycleGAN = cycleGAN256()
cycleGAN.built = True
cycleGAN.load_weights(checkpoint_path)

def TF2CV(im):
    img = tf.cast(tf.math.scalar_mul(255/2, im[0]+1), dtype=tf.uint8)
    img_ = np.array(tf.keras.utils.array_to_img(img),dtype='uint8')
    img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
    return img_

def histEqu(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output

def get_prediction(model, test_input):   #MAY BE REMOVED 
    prediction = model(test_input, training=True)
    return prediction

def processImage(indir, outdir, fname, mode):
    image = cv2.imread(f'{indir}/{fname}', cv2.IMREAD_COLOR)
    _, w, _ = np.shape(image)
    w = w // 3
    #he = histEqu(image[:, :w, :])
    he = image[:, :w, :]
    mt = image[:, w:(2*w), :]
    virtual = image[:, (2*w):, :]
    he_ = np.expand_dims((he-255/2)/(255/2), axis=0)
    if mode=='HE':
        im = he
        gt = MT2F_CLR(mt)
    elif mode =='cGAN':
        im = virtual
        gt = MT2F_CLR(mt)
        #im = TF2CV(im)
    elif mode == 'cycleGAN':
        im = virtual
        gt = MT2F_CLR(mt)
        im = get_prediction(cycleGAN.generator_g, he_)
        im = TF2CV(im)
    output = cv2.hconcat((im, gt))
    cv2.imwrite(f'{outdir}/{fname}', output)
    return 

""" parser = argparse.ArgumentParser(description='Naglah Segmentation Metrics')
parser.add_argument("--dataroot", required= True, help="root directory that contains the data")
parser.add_argument("--experiment_id", required= True, type=str, help="Experiment ID to track experiment and results" )

params = parser.parse_args()

PATH = params.dataroot
EXPERIMENT_ID = params.experiment_id """

PATH = "F:/N002-Research/liver-pathology/segmentation/validation_new"
EXPERIMENT_ID = 'PREPARE-SEGMENTATION-DATASET-Validation-New'

input_subdir = 'raw/'
output_subdirHE = 'input/HE'
output_subdirMT_cGAN = 'input/MT_cGAN'
output_subdirMT_cycleGAN = 'input/MT_cycleGAN'

createDir(f'{PATH}/input')
createDir(f'{PATH}/{output_subdirHE}')
createDir(f'{PATH}/{output_subdirMT_cGAN}')
createDir(f'{PATH}/{output_subdirMT_cycleGAN}')

# Load HE Images
datadir = f'{PATH}/{input_subdir}/monitor_output_cGAN'
outdir = f'{PATH}/{output_subdirHE}'
mode = 'HE'

fnames = os.listdir(datadir)
for i in range(len(fnames)):
    fname = fnames[i]
    processImage(datadir, outdir, fname, mode)
    if i%1000==0:
        print(f"Processing {i}")

# Load cGAN Images
datadir = f'{PATH}/{input_subdir}/monitor_output_cGAN'
outdir = f'{PATH}/{output_subdirMT_cGAN}'
mode = 'cGAN'

fnames = os.listdir(datadir)
for i in range(len(fnames)):
    fname = fnames[i]
    processImage(datadir, outdir, fname, mode)
    if i%1000==0:
        print(f"Processing {i}")

# Load cycleGAN Images
datadir = f'{PATH}/{input_subdir}/monitor_output_cycleGAN'
outdir = f'{PATH}/{output_subdirMT_cycleGAN}'
mode = 'cycleGAN'

fnames = os.listdir(datadir)
for i in range(len(fnames)):
    fname = fnames[i]
    processImage(datadir, outdir, fname, mode)
    if i%1000==0:
        print(f"Processing {i}")

print(f"Done, Experiment ID:{EXPERIMENT_ID}")