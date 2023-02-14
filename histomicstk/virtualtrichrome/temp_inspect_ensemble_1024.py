import os
import cv2
import numpy as np

PATH = "F:/N002-Research/liver-pathology/liver_ensemble/liver_cycleGAN1024"

sub = 'trainA_'

fnames = os.listdir(f"{PATH}/{sub}")

def createDir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False 

createDir(f"{PATH}/{sub}_")

for fname in fnames:
    try:
        im = cv2.imread(f"{PATH}/{sub}/{fname}", cv2.IMREAD_COLOR)
        h, w, _ = np.shape(im)
        if h==1024 and w==1024:
            cv2.imwrite(f"{PATH}/{sub}_/{fname}", im)
        else:
            print(f"Error in: {fname}")
    except:
        print(f"Read Error in: {fname}")


sub = 'trainB_'

fnames = os.listdir(f"{PATH}/{sub}")

createDir(f"{PATH}/{sub}_")

for fname in fnames:
    try:
        im = cv2.imread(f"{PATH}/{sub}/{fname}", cv2.IMREAD_COLOR)
        h, w, _ = np.shape(im)
        if h==1024 and w==1024:
            cv2.imwrite(f"{PATH}/{sub}_/{fname}", im)
        else:
            print(f"Error in: {fname}")
    except:
        print(f"Read Error in: {fname}")
