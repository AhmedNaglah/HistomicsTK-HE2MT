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

def load_triple(image_file):

  image = cv2.imread(image_file, cv2.IMREAD_COLOR)
  _, w, _ = np.shape(image)
  w = w // 3

  input_image = image[:, :w, :]
  gt = image[:, w:(2*w), :]

  prediction = MT2F_CLR(input_image)

  return input_image, gt, prediction

def load_image_segmentation(image_file):
  input_image, gt, prediction  = load_triple(image_file)

  return input_image, gt, prediction

def isTargetImage(mask, thre):
    mask = mask>128
    print(np.sum(mask)/(mask.size))
    return np.sum(mask)/(mask.size)> thre
    

""" parser = argparse.ArgumentParser(description='Naglah Segmentation Metrics')
parser.add_argument("--dataroot", required= True, help="root directory that contains the data")
parser.add_argument("--experiment_id", required= True, type=str, help="Experiment ID to track experiment and results" )

params = parser.parse_args()

PATH = params.dataroot
EXPERIMENT_ID = params.experiment_id """

PATH = "F:/N002-Research/liver-pathology/fibrosis_segmentation"
EXPERIMENT_ID = 'GET_TARGET_IMAGE_70'

# Load Images at directory
datadir = f'{PATH}/output'
fnames = os.listdir(datadir)

f = open(f'{PATH}/{EXPERIMENT_ID}_targetimages.out', 'w')

for i in range(len(fnames)):
    fname = fnames[i]
    input_image, gt, prediction = load_image_segmentation(f'{datadir}/{fname}')
    if isTargetImage(gt, 0.75): print( f"{fname}", file=f)
    if i%1000==0:
        print(f"processing {i}")
f.close()

print(f"Done, Experiment ID:{EXPERIMENT_ID}")