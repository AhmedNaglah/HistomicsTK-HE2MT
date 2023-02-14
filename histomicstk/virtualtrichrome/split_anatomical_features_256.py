import cv2
import tensorflow as tf
import argparse
import importlib
import os
import numpy as np
import pandas as pd
import shutil

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

def createDirectories(PATH, tftr):
    createDir(f"{PATH}/{tftr}")
    createDir(f"{PATH}/{tftr}/input_256")
    createDir(f"{PATH}/{tftr}/output_256")
    ftr_map = []
    sub_input = os.listdir(f"{PATH}/input_256")
    for d in sub_input:
        createDir(f"{PATH}/{tftr}/input_256/{d}")
        ftr_map.append(  (f"{PATH}/input_256/{d}", f"{PATH}/{tftr}/input_256/{d}")  )
    sub_input = os.listdir(f"{PATH}/output_256")
    for d in sub_input:
        createDir(f"{PATH}/{tftr}/output_256/{d}")
        ftr_map.append(  (f"{PATH}/output_256/{d}", f"{PATH}/{tftr}/output_256/{d}")  )
    return ftr_map

def copyFiles(fnames, ds):
    for fname in fnames:
        for d in ds:
            src, dst = d
            fname_a = fname.replace(".png", "_a.png")
            fname_b = fname.replace(".png", "_b.png")
            fname_c = fname.replace(".png", "_c.png")
            fname_d = fname.replace(".png", "_d.png")
            shutil.copy(f'{src}/{fname_a}', f'{dst}/{fname_a}')
            shutil.copy(f'{src}/{fname_b}', f'{dst}/{fname_b}')
            shutil.copy(f'{src}/{fname_c}', f'{dst}/{fname_c}')
            shutil.copy(f'{src}/{fname_d}', f'{dst}/{fname_d}')
    
PATH = "F:/N002-Research/liver-pathology/segmentation/anatomical_features"
EXPERIMENT_ID = 'ANATOMICAL-FEATURES-SPLIT' 

f = open(f"{PATH}/features.txt", 'r')

ftr_map = list(map(lambda x: x.replace("\n","").split(";"), f))

target_ftrs = ["portal", "hepatic", "bile"]

for tftr in target_ftrs:
    dir_map = createDirectories(PATH, tftr)
    fnames = [x[0] for x in ftr_map if x[1]==tftr]
    copyFiles(fnames, dir_map)
