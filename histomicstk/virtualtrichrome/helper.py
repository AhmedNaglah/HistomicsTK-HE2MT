import openslide
import numpy as np
import cv2
import tensorflow as tf

def predictWSI(wsifile, model):
    s =  openslide(wsifile)

def WSI2CSV(im):
    im = np.array(im.convert("RGB"))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im

def TF2CV(im):
    img = tf.cast(tf.math.scalar_mul(255/2, im[0]+1), dtype=tf.uint8)
    img_ = np.array(tf.keras.preprocessing.image.array_to_img(img),dtype='uint8')
    img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
    return img_

def CV2TF(im):
    im = (im-(255/2))/(255/2)
    rgb_tensor = tf.convert_to_tensor(im, dtype=tf.float32)
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    return rgb_tensor

def AdaptBeforePredict(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.array(im, dtype='float32')
    im = np.expand_dims(im, axis=0)
    im = (im/(255/2)) - 1.0
    return im

def AdaptPredicted(im):
    im = np.squeeze(im)
    im = (im + 1.0)*(255/2)
    im = np.array(im, dtype='uint8')
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im
