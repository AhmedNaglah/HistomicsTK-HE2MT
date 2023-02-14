import cv2
import tensorflow as tf
import argparse
import importlib
import os
import numpy as np
import pandas as pd

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

exps = [("F:/N002-Research/liver-pathology/segmentation_he2fibrosis_", 'HE2F-75'),("F:/N002-Research/liver-pathology/fibrosis_segmentation", 'MT2F-UNET-75'), ("F:/N002-Research/liver-pathology/MT2F_CLR_RESULTS", 'MT2F-CLR-75')]

for exp in exps:
    PATH, EXPERIMENT_ID = exp

    """ PATH = "F:/N002-Research/liver-pathology/segmentation_he2fibrosis_"
    PATH = "F:/N002-Research/liver-pathology/fibrosis_segmentation"
    PATH = "F:/N002-Research/liver-pathology/MT2F_CLR_RESULTS"
    EXPERIMENT_ID = 'MT2F-CLR' """
    TARGET_PATCHES = 'F:/N002-Research/liver-pathology/fibrosis_segmentation/GET_TARGET_IMAGE_70_targetimages.out'

    # Load Target Images
    f2 = open(f"{TARGET_PATCHES}", "r")
    target_images = []
    for x in f2:
        target_images.append(x.replace('\n', ''))

    # Load Images at directory
    datadir = f'{PATH}/output'
    fnames = os.listdir(datadir)

    f = open(f'{PATH}/{EXPERIMENT_ID}.out', 'w')
    f_ = open(f'{PATH}/{EXPERIMENT_ID}_targetImages.out', 'w')

    cols = ['fname', 'acc', 'iou', 'iouA', 'iouB', 'dice', 'dice2']
    df = pd.DataFrame(columns=cols)

    for i in range(len(fnames)):
        fname = fnames[i]
        input_image, gt, prediction = load_image_segmentation(f'{datadir}/{fname}')
        acc, iou, iouA, iouB, dice, dice2 = segmentation_metrics(gt, prediction)
        row = {'fname':fname, 'acc':acc, 'iou':iou, 'iouA':iouA, 'iouB':iouB, 'dice':dice, 'dice2':dice2}
        df = df.append(row, ignore_index=True)
        if i%1000==0:
            print(f"Processing {i}")
        print(f'{fname};{acc};{iou};{iouA};{iouB};{dice};{dice2}', file=f)

    df_summary = df.loc[df['fname'].isin(target_images)]

    for index, row in df_summary.iterrows():
        fname = row['fname']
        acc = row['acc']
        iou = row['iou']
        iouA = row['iouA']
        iouB = row['iouB']
        dice = row['dice']
        dice2 = row['dice2']
        print(f'{fname};{acc};{iou};{iouA};{iouB};{dice};{dice2}', file=f_)

    f3 = open(f'{PATH}/{EXPERIMENT_ID}_summary.out', 'w')
    print(f'Experiment Summary: {EXPERIMENT_ID}', file=f3)
    print(f'-----------------------------------', file=f3)
    print(df_summary.mean(), file=f3)
    print(df_summary.std(), file=f3)
    print(f'-----------------------------------', file=f3)
    print(df_summary.describe(), file=f3)

    print(f"Done, Experiment ID:{EXPERIMENT_ID}")