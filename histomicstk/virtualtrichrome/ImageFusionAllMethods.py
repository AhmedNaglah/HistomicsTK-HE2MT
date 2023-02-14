import cv2
import os
import numpy as np
from fusion_methods import *

def createDir(mydir):
    if not os.path.exists(mydir):
        os.makedirs(mydir)

root = "F:/N002-Research/liver-pathology/mia2"

sizes = [
    "liver64_256tiles",
    "liver128_256tiles",
    "liver256_256tiles",
    "liver512_256tiles",
    "liver1024_256tiles",
]

# Loop Images
gt_fub_folder = "gt"
fnames = os.listdir(f"{root}/{gt_fub_folder}")


method = 'ExposureFusion'
createDir(f"{root}/{method}")

for fname in fnames:
    try:
        images = []
        fname_ = fname.replace("real_B", "fake_B")
        for s in sizes:
            im = cv2.imread(f"{root}/{s}/{fname_}", cv2.IMREAD_COLOR)
            if im is not None:
                images.append(im)
        fused_image = ExposureFusion(images)
        fname__ = fname.replace("real_B", "fused_B")
        cv2.imwrite(f"{root}/{method}/{fname__}", fused_image)
    except:
        print(f'error {fname}')


method = 'DWTNImages'
createDir(f"{root}/{method}")

for fname in fnames:
    try:
        images = []
        fname_ = fname.replace("real_B", "fake_B")
        for s in sizes:
            im = cv2.imread(f"{root}/{s}/{fname_}", cv2.IMREAD_COLOR)
            if im is not None:
                images.append(im)
            fused_image = DWTNImages(images)
        fname__ = fname.replace("real_B", "fused_B")
        cv2.imwrite(f"{root}/{method}/{fname__}", fused_image)
    except:
        print(f'error {fname}')

