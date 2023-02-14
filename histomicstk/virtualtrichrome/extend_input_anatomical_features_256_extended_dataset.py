from re import M
import cv2
import tensorflow as tf
import argparse
import importlib
import os
import numpy as np
import pandas as pd
from tensorflow import keras

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
    im = np.array(input_image[0]*255, dtype='uint8')
    roi_hue = [170, 260]
    roi_hue_255 = [int(round(k*255/360,0)) for k in roi_hue]
    roi_hue_255 = [110, 130]

    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    low_blue = (roi_hue_255[0],50,50)
    high_blue = (roi_hue_255[1],255,255)

    blue_mask = cv2.inRange(im_hsv, low_blue, high_blue)
    blue_gt = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
    return blue_gt

""" parser = argparse.ArgumentParser(description='Naglah Segmentation Metrics')
parser.add_argument("--dataroot", required= True, help="root directory that contains the data")
parser.add_argument("--experiment_id", required= True, type=str, help="Experiment ID to track experiment and results" )

params = parser.parse_args()

PATH = params.dataroot
EXPERIMENT_ID = params.experiment_id """

""" # HE2MT
modelname = "condGAN"
cGAN = tf.keras.models.load_model(f"{PATH}/{modelname}.h5")
modelname = "cycleGAN"
cycleGAN = tf.keras.models.load_model(f"{PATH}/{modelname}.h5") """

""" PATH = "F:/N002-Research/liver-pathology/segmentation_he2fibrosis_"
PATH = "F:/N002-Research/liver-pathology/fibrosis_segmentation"
PATH = "F:/N002-Research/liver-pathology/MT2F_CLR_RESULTS" """

ROOT = "F:/N002-Research/liver-pathology/segmentation/extended_anatomical_each"

ftrs = [ "bile", "hepatic", "portal"]

for ftr in ftrs:

    PATH = f"{ROOT}/{ftr}"
    EXPERIMENT_ID = 'ANATOMICAL-FEATURES_ftrs_extended' 

    # Load Images at directory
    datadirHE = f'{PATH}/input/HE'
    datadircGAN = f'{PATH}/input/MT_cGAN'
    datadircycleGAN = f'{PATH}/input/MT_cycleGAN'

    outdirHE = f'{PATH}/output_256/HE'
    outdircGAN_MT2F_CLR = f'{PATH}/output_256/cGAN_MT2F_CLR'
    outdircGAN_MT2F_UNET = f'{PATH}/output_256/cGAN_MT2F_UNET'
    outdircycleGAN_MT2F_CLR = f'{PATH}/output_256/cycleGAN_MT2F_CLR'
    outdircycleGAN_MT2F_UNET = f'{PATH}/output_256/cycleGAN_MT2F_UNET'

    createDir(f'{PATH}/input_256')
    createDir(f'{PATH}/input_256/MT_cGAN')
    createDir(f'{PATH}/input_256/MT_cycleGAN')
    createDir(f'{PATH}/input_256/HE')

    createDir(f'{PATH}/output_256')
    createDir(outdirHE)
    createDir(outdircGAN_MT2F_CLR)
    createDir(outdircGAN_MT2F_UNET)
    createDir(outdircycleGAN_MT2F_CLR)
    createDir(outdircycleGAN_MT2F_UNET)


    def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
        return c, p

    def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
        us = keras.layers.UpSampling2D((2, 2))(x)
        concat = keras.layers.Concatenate()([us, skip])
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        return c

    def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        return c

    def UNet():
        f = [16, 16, 32, 64, 128, 256]
        inputs = keras.layers.Input((256, 256, 3))
        
        p0 = inputs
        c1, p1 = down_block(p0, f[0]) #128 -> 64
        c2, p2 = down_block(p1, f[1]) #64 -> 32
        c3, p3 = down_block(p2, f[2]) #32 -> 16
        c4, p4 = down_block(p3, f[3]) #16->8
        c5, p5 = down_block(p4, f[4]) #16->8

        bn = bottleneck(p5, f[5])

        u0 = up_block(bn, c5, f[4]) #8 -> 16
        u1 = up_block(u0, c4, f[3]) #8 -> 16
        u2 = up_block(u1, c3, f[2]) #16 -> 32
        u3 = up_block(u2, c2, f[1]) #32 -> 64
        u4 = up_block(u3, c1, f[0]) #64 -> 128
        
        outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
        model = keras.models.Model(inputs, outputs)
        return model

    HE2F = UNet()
    HE2F.load_weights("F:/N002-Research/liver-pathology/segmentation/transformation_models/HE2F.h5")

    UNET = UNet()
    UNET.load_weights("F:/N002-Research/liver-pathology/segmentation/transformation_models/MT2F-UNET.h5")

    def TF2CV(im):
        img = tf.cast(tf.math.scalar_mul(255/2, im[0]+1), dtype=tf.uint8)
        img_ = np.array(keras.utils.array_to_img(img),dtype='uint8')
        img_ = cv2.cvtColor(img_, cv2.COLOR_GRAY2BGR)
        return img_

    def getMT(im_, modelname):
        global HE2F
        im2 = np.expand_dims(im_, axis=0)/255
        if modelname=="HE2F":
            out = HE2F(im2)
            return TF2CV(out)
        elif modelname=="condGAN":
            out = MT2F_CLR(im2)
        elif modelname=="cycleGAN":
            out = MT2F_CLR(im2)
        elif modelname=="cycleGAN_UNET":
            out = UNET(im2)
            return TF2CV(out)
        elif modelname=="cGAN_UNET":
            out = UNET(im2)
            return TF2CV(out)
        return out

    def process_image(indir, outdir, modelname, fname):
        fname_a = fname.replace(".png", "_a.png")
        fname_b = fname.replace(".png", "_b.png")
        fname_c = fname.replace(".png", "_c.png")
        fname_d = fname.replace(".png", "_d.png")

        image = cv2.imread(f"{indir}/{fname}", cv2.IMREAD_COLOR)
        im = image[:, :512, :]
        gt = image[:, 512:, :]
        a = im[0:256, 0:256, :]
        b = im[0:256, 256:, :]
        c = im[256:, 0:256, :]
        d = im[256:, 256:, :]
        gt_a = gt[0:256, 0:256, :]
        gt_b = gt[0:256, 256:, :]
        gt_c = gt[256:, 0:256, :]
        gt_d = gt[256:, 256:, :]

        out_a = cv2.hconcat((a, gt_a))
        out_b = cv2.hconcat((b, gt_b))
        out_c = cv2.hconcat((c, gt_c))
        out_d = cv2.hconcat((d, gt_d))
        indir_new = indir.replace("input", "input_256")

        cv2.imwrite(f"{indir_new}/{fname_a}", out_a)
        cv2.imwrite(f"{indir_new}/{fname_b}", out_b)
        cv2.imwrite(f"{indir_new}/{fname_c}", out_c)
        cv2.imwrite(f"{indir_new}/{fname_d}", out_d)
        a_ = getMT(a, modelname)
        b_ = getMT(b, modelname)
        c_ = getMT(c, modelname)
        d_ = getMT(d, modelname)
        out_a = cv2.hconcat((a, gt_a, a_))
        out_b = cv2.hconcat((b, gt_b, b_))
        out_c = cv2.hconcat((c, gt_c, c_))
        out_d = cv2.hconcat((d, gt_d, d_))

        cv2.imwrite(f"{outdir}/{fname_a}", out_a)
        cv2.imwrite(f"{outdir}/{fname_b}", out_b)
        cv2.imwrite(f"{outdir}/{fname_c}", out_c)
        cv2.imwrite(f"{outdir}/{fname_d}", out_d)

    def process_image_(indir, outdir, modelname, fname):
        image = cv2.imread(f"{indir}/{fname}", cv2.IMREAD_COLOR)
        im = image[:, :512, :]
        gt = image[:, 512:, :]
        a = im[0:256, 0:256, :]
        b = im[0:256, 256:, :]
        c = im[256:, 0:256, :]
        d = im[256:, 256:, :]
        gt_a = gt[0:256, 0:256, :]
        gt_b = gt[0:256, 256:, :]
        gt_c = gt[256:, 0:256, :]
        gt_d = gt[256:, 256:, :]
        a_ = getMT(a, modelname)
        b_ = getMT(b, modelname)
        c_ = getMT(c, modelname)
        d_ = getMT(d, modelname)
        out_a = cv2.hconcat((a_, gt_a))
        out_b = cv2.hconcat((b_, gt_b))
        out_c = cv2.hconcat((c_, gt_c))
        out_d = cv2.hconcat((d_, gt_d))
        fname_a = fname.replace(".png", "_a.png")
        fname_b = fname.replace(".png", "_b.png")
        fname_c = fname.replace(".png", "_c.png")
        fname_d = fname.replace(".png", "_d.png")

        cv2.imwrite(f"{outdir}/{fname_a}", out_a)
        cv2.imwrite(f"{outdir}/{fname_b}", out_b)
        cv2.imwrite(f"{outdir}/{fname_c}", out_c)
        cv2.imwrite(f"{outdir}/{fname_d}", out_d)

    modelname = 'HE2F'
    indir = datadirHE
    outdir = outdirHE

    fnames = os.listdir(indir)
    for i in range(len(fnames)):
        fname = fnames[i]
        patches = process_image(indir, outdir, modelname, fname)


    modelname = 'condGAN'
    indir = datadircGAN
    outdir = outdircGAN_MT2F_CLR

    fnames = os.listdir(indir)
    for i in range(len(fnames)):
        fname = fnames[i]
        patches = process_image(indir, outdir, modelname, fname)


    modelname = 'cGAN_UNET'
    indir = datadircGAN
    outdir = outdircGAN_MT2F_UNET

    fnames = os.listdir(indir)
    for i in range(len(fnames)):
        fname = fnames[i]
        patches = process_image(indir, outdir, modelname, fname)


    modelname = 'cycleGAN'
    indir = datadircycleGAN
    outdir = outdircycleGAN_MT2F_CLR

    fnames = os.listdir(indir)
    for i in range(len(fnames)):
        fname = fnames[i]
        patches = process_image(indir, outdir, modelname, fname)


    modelname = 'cycleGAN_UNET'
    indir = datadircycleGAN
    outdir = outdircycleGAN_MT2F_UNET

    fnames = os.listdir(indir)
    for i in range(len(fnames)):
        fname = fnames[i]
        patches = process_image(indir, outdir, modelname, fname)



