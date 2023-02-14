import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = "C:/Users/inagl/Downloads/MedIA - Conditional GANs-based fibrosis detection and quantification in Hematoxylin and Eosin whole slide images - V4"
SUBDIR = "images"

OUTSUBDIR = "images_eps"

def createDir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False 

createDir(f"{PATH}/{OUTSUBDIR}")

fnames = os.listdir(f"{PATH}/{SUBDIR}")

h_ = 1024

for fname in fnames:
    im = cv2.imread(f"{PATH}/{SUBDIR}/{fname}", cv2.IMREAD_COLOR)
    fname_jpg = fname.replace(".png", ".jpg")
    h, w, _ = np.shape(im)
    w_ = int(h_/h * w)
    im = cv2.resize(im, (w_, h_))
    cv2.imwrite(f"{PATH}/{OUTSUBDIR}/{fname_jpg}", im)

