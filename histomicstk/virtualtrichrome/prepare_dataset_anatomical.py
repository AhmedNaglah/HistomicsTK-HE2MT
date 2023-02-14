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

def MT2F_CLR(input_image):
    roi_hue = [170, 260]
    roi_hue_255 = [int(round(k*255/360,0)) for k in roi_hue]
    roi_hue_255 = [110, 130]

    im_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

    low_blue = (roi_hue_255[0],50,50)
    high_blue = (roi_hue_255[1],255,255)

    blue_mask = cv2.inRange(im_hsv, low_blue, high_blue)
    blue_gt = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
    return blue_gt

def processImage(indir, outdir, fname):
    indircGAN, indircycleGAN = indir
    outdirHE, outdirMTcGAN, outdirMTcycleGAN = outdir
    if fname.endswith("real_B.png"):
        image = cv2.imread(f'{indircGAN}/{fname}', cv2.IMREAD_COLOR)
        gt = MT2F_CLR(image)
        fname2 = fname.replace("real_B", "real_A")
        he = cv2.imread(f'{indircGAN}/{fname2}', cv2.IMREAD_COLOR)
        fname3 = fname.replace("real_B", "fake_B")
        cGAN = cv2.imread(f'{indircGAN}/{fname3}', cv2.IMREAD_COLOR)
        cycleGAN = cv2.imread(f'{indircycleGAN}/{fname3}', cv2.IMREAD_COLOR)
        outhe = cv2.hconcat((he, gt))
        outcGAN = cv2.hconcat((cGAN, gt))
        outcycleGAN = cv2.hconcat((cycleGAN, gt))

        cv2.imwrite( f"{outdirHE}/{fname}", outhe)
        cv2.imwrite( f"{outdirMTcGAN}/{fname}", outcGAN)
        cv2.imwrite( f"{outdirMTcycleGAN}/{fname}", outcycleGAN)

    return 


""" parser = argparse.ArgumentParser(description='Naglah Segmentation Metrics')
parser.add_argument("--dataroot", required= True, help="root directory that contains the data")
parser.add_argument("--experiment_id", required= True, type=str, help="Experiment ID to track experiment and results" )

params = parser.parse_args()

PATH = params.dataroot
EXPERIMENT_ID = params.experiment_id """

PATH = "F:/N002-Research/liver-pathology/segmentation/anatomical_features"
EXPERIMENT_ID = 'PREPARE-SEGMENTATION-DATASET-ANATOMICAL'

input_subdir_cGAN = f'{PATH}/input/cGAN'
input_subdir_cycleGAN = f'{PATH}/input/cycleGAN'

output_subdir = f'{PATH}/input_'
createDir(output_subdir)

out_he = f"{output_subdir}/HE"
out_mtcGAN = f"{output_subdir}/MT_cGAN"
out_mtcycleGAN = f"{output_subdir}/MT_cycleGAN"

createDir(out_he)
createDir(out_mtcGAN)
createDir(out_mtcycleGAN)

fnames = os.listdir(input_subdir_cGAN)
for i in range(len(fnames)):
    fname = fnames[i]
    processImage((input_subdir_cGAN, input_subdir_cycleGAN), (out_he, out_mtcGAN, out_mtcycleGAN), fname)
    print(f"Processing {i}")

