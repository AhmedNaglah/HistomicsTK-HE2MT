import cv2
import tensorflow as tf
import argparse
import importlib
import os
import numpy as np
import pandas as pd

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

""" parser = argparse.ArgumentParser(description='Naglah Segmentation Metrics')
parser.add_argument("--dataroot", required= True, help="root directory that contains the data")
parser.add_argument("--experiment_id", required= True, type=str, help="Experiment ID to track experiment and results" )

params = parser.parse_args()

PATH = params.dataroot
EXPERIMENT_ID = params.experiment_id """

""" PATH = "F:/N002-Research/liver-pathology/segmentation_he2fibrosis_"
PATH = "F:/N002-Research/liver-pathology/fibrosis_segmentation"
PATH = "F:/N002-Research/liver-pathology/MT2F_CLR_RESULTS" """

PATH = "C:/Users/inagl/Desktop/mia/raw_fig/semantic_review"
EXPERIMENT_ID = 'ANATOMICAL-FEATURES' 

# Load Images at directory
datadir = f'{PATH}/input'
fnames = os.listdir(datadir)

# HE2MT
modelname = "condGAN"
cGAN = tf.keras.models.load_model(f"{PATH}/{modelname}.h5")
modelname = "cycleGAN"
cycleGAN = tf.keras.models.load_model(f"{PATH}/{modelname}.h5")

def getTiles(im):
    pass

def stackTiles(patches):
    pass

def getMT(im, modelname):
    global cGAN
    global cycleGAN
    if modelname=="condGAN":
        out = cGAN(im)
    elif modelname=="cycleGAN":
        out = cycleGAN(im)
    return out

modelname = "condGAN"
createDir(f"{PATH}/{modelname}_prediction")

for i in range(len(fnames)):
    fname = fnames[i]
    if fname.endswith('real_A.png'):
        im = cv2.imread(f"{PATH}/input/{fname}", cv2.IMREAD_COLOR)
        patches = getTiles(im)
        patches_mt = [None for _ in range(len(patches))]
        for j in range(len(patches)):
            patches_mt[j] = getMT(patches[j], modelname)
        im_out = stackTiles(patches_mt)


