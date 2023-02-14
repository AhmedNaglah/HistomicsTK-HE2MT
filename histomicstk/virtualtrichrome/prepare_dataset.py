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

def processImage(indir, outdir, fname):
    image = cv2.imread(f'{indir}/{fname}', cv2.IMREAD_COLOR)
    _, w, _ = np.shape(image)
    w = w // 3
    image = image[:, :2*w, :]
    cv2.imwrite(f'{outdir}/{fname}', image)
    return 



""" parser = argparse.ArgumentParser(description='Naglah Segmentation Metrics')
parser.add_argument("--dataroot", required= True, help="root directory that contains the data")
parser.add_argument("--experiment_id", required= True, type=str, help="Experiment ID to track experiment and results" )

params = parser.parse_args()

PATH = params.dataroot
EXPERIMENT_ID = params.experiment_id """

PATH = "F:/N002-Research/liver-pathology/segmentation/generic_samples"
EXPERIMENT_ID = 'PREPARE-SEGMENTATION-DATASET'

input_subdirHE = 'output/HE2F/visualization'
input_subdirMT = 'output/MT2F-CLR/visualization'
output_subdirHE = 'input/HE'
output_subdirMT = 'input/MT'

createDir(f'{PATH}/{output_subdirHE}')
createDir(f'{PATH}/{output_subdirMT}')

# Load HE Images
datadir = f'{PATH}/{input_subdirHE}'
outdir = f'{PATH}/{output_subdirHE}'

fnames = os.listdir(datadir)
for i in range(len(fnames)):
    fname = fnames[i]
    processImage(datadir, outdir, fname)
    if i%1000==0:
        print(f"Processing {i}")

# Load HE Images
datadir = f'{PATH}/{input_subdirMT}'
outdir = f'{PATH}/{output_subdirMT}'

fnames = os.listdir(datadir)
for i in range(len(fnames)):
    fname = fnames[i]
    processImage(datadir, outdir, fname)
    if i%1000==0:
        print(f"Processing {i}")

print(f"Done, Experiment ID:{EXPERIMENT_ID}")