import importlib
import os
import tensorflow as tf
import numpy as np
import cv2
from models.condGAN256 import condGAN256
from models.cycleGAN256 import cycleGAN256

import importlib

DATAROOT = "F:/N002-Research/liver-pathology/segmentation"

checkpoint_path = "F:/N002-Research/liver-pathology/segmentation/transformation_models/cGAN/ckpt-14"
checkpoint_path = "D:/codes/media/cGAN_checkpoint_fold0/training_checkpoints/ckpt-11"
cGAN = condGAN256()
cGAN.built = True
cGAN.restore_from_checkpoint(checkpoint_path)

checkpoint_path = "F:/N002-Research/liver-pathology/segmentation/transformation_models/cycleGAN/ckpt-7"
cycleGAN = cycleGAN256()
cycleGAN.built = True
cycleGAN.load_weights(checkpoint_path)

print('HERE')

def createDir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

def load_double(image_file):

  image = cv2.imread(image_file, cv2.IMREAD_COLOR)
  _, w, _= np.shape(image)
  w = w // 2

  he = image[:, :w, :]
  mt = image[:, w:, :]


  return he, mt

def load_image(image_file):
  he, mt  = load_double(image_file)

  return he, mt

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

sample_image_path = "F:/N002-Research/liver-pathology/segmentation/validation_samples/raw/101_201_(61796, 46080)_(64177, 46045).jpg"

he, mt = load_image(sample_image_path)
he_, mt_ = normalize(he, mt)
he_ = np.expand_dims(he_, axis=0)

mt_virtual = cycleGAN(he_)

mt_virtual_ = TF2CV(mt_virtual)

out = cv2.hconcat((he, mt, mt_virtual_))

cv2.imshow('Prediction Sample', out)
cv2.waitKey(0)

print('HERE')
