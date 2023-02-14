import cv2
from numpy.lib.type_check import imag
import tensorflow as tf
import argparse
import os
import numpy as np
import pandas as pd
import math

def createDir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False 

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

def mutual_information(hgram):
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def nmi_evaluate(hgram):
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x

    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    nzx = px > 0 # Only non-zero pxy values contribute to the sum
    nzy = py > 0 # Only non-zero pxy values contribute to the sum

    Hx = np.sum(-px[nzx]*np.log(px[nzx]))
    Hy = np.sum(-py[nzy]*np.log(py[nzy]))
    nmi =  2*(np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs])))/(Hx+Hy)
    if nmi==None:
        return -1
    else:
        return nmi

def image_similarity(self, im1, im2, d=256):

    im1 = self.TF2CV(im1)
    im2 = self.TF2CV(im2)
    imBr_ = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)        
    imBg_ = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)

    hBr = imBr_[:,:,0]
    hBg = imBg_[:,:,0]
    hist_2d, x_edges, y_edges = np.histogram2d(
        hBr.ravel(),
        hBg.ravel(),
        bins=d)
    mi = mutual_information(hist_2d)

    h = hBr.flatten()
    t = hBg.flatten()

    h = hBr.ravel()
    t = hBg.ravel()

    h2d_ht, _, _ = np.histogram2d(h.ravel(), t.ravel(), bins=d, normed=True)

    nmi = nmi_evaluate(h2d_ht)

    return mi, nmi

def getMI(im1, im2):
    d= 256

    imBr_ = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)        
    imBg_ = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)

    hBr = imBr_[:,:,0]
    hBg = imBg_[:,:,0]
    hist_2d, x_edges, y_edges = np.histogram2d(
        hBr.ravel(),
        hBg.ravel(),
        bins=d)
    mi = mutual_information(hist_2d)
    return mi

def getNMI(im1, im2):
    d= 256

    imBr_ = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)        
    imBg_ = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)

    hBr = imBr_[:,:,0]
    hBg = imBg_[:,:,0]
    h = hBr.flatten()
    t = hBg.flatten()


    h = hBr.ravel()
    t = hBg.ravel()

    h2d_ht, _, _ = np.histogram2d(h.ravel(), t.ravel(), bins=d, normed=True)

    nmi = nmi_evaluate(h2d_ht)
    return nmi

def getHC(im1, im2):
    d= 256

    method = cv2.HISTCMP_CORREL
    imBr_ = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)        
    imBg_ = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)
    hBr = imBr_[:,:,0]
    hBg = imBg_[:,:,0]
    histBr,bins = np.histogram(hBr.ravel(),d,[0,d])
    histBg,bins = np.histogram(hBg.ravel(),d,[0,d])
    hc = cv2.compareHist(histBr.astype(dtype=np.float32), histBg.astype(dtype=np.float32), method )
    return hc

def getBCD(im1, im2):
    d= 256
    method = cv2.HISTCMP_BHATTACHARYYA
    imBr_ = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)        
    imBg_ = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)
    hBr = imBr_[:,:,0]
    hBg = imBg_[:,:,0]
    histBr,bins = np.histogram(hBr.ravel(),d,[0,d])
    histBg,bins = np.histogram(hBg.ravel(),d,[0,d])
    bcd = cv2.compareHist(histBr.astype(dtype=np.float32), histBg.astype(dtype=np.float32), method )
    return bcd

def similarity_metrics(im1,im2):
    mi = getMI(im1, im2)
    nmi = getNMI(im1, im2)
    hc = getHC(im1, im2)
    bcd = getBCD(im1, im2)

    return mi,nmi,hc,bcd

def load_double(image_file):

  image = cv2.imread(image_file, cv2.IMREAD_COLOR)
  _, w, _ = np.shape(image)
  w = w // 2

  input_image = image[:, :w, :]
  prediction = image[:, w:, :]

  return input_image, prediction

def load_image_similarity(image_file):
  input_image, prediction  = load_double(image_file)

  return input_image, prediction

parser = argparse.ArgumentParser(description='Naglah Segmentation Metrics')
parser.add_argument("--dataroot", required= True, help="root directory that contains the data")
parser.add_argument("--experiment_id", required= True, type=str, help="Experiment ID to track experiment and results" )
parser.add_argument("--ensemble", required= True, type=str, help="Experiment ID to track experiment and results" )

params = parser.parse_args()

PATH = params.dataroot
EXPERIMENT_ID = params.experiment_id
ENSEMBLE = params.ensemble

""" PATH = "F:/N002-Research/liver-pathology/liver_ensemble/"
EXPERIMENT_ID = 'similarity_trial'
ENSEMBLE = 'D:\codes\media2\process_ensemble_cGAN.txt'
 """

createDir(f"{PATH}/{EXPERIMENT_ID}")

ensemble = open(ENSEMBLE)
others = []
for elem in ensemble:
    folder, model, subfolder = elem.replace("\n", "").split(',')

    if folder.endswith('1024'):
        reference = (folder, model, subfolder)
    else:
        others.append((folder, model, subfolder))

folder_ref, model_ref, subfolder_ref = reference

fnames = os.listdir(f"{PATH}/{folder_ref}/{subfolder_ref}/prediction_results")

def getStackedImage(stacked_image_meta):
    fnames, folder = stacked_image_meta
    l = len(fnames)
    base = fnames[0].split("/")[-1].replace("\n", "")

    def concat_tile(im_list_2d):
        return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])
    im_tiles = [[cv2.imread(f"{folder}/{base}_{i+j}.jpg") for i in range(l)] for j in range(l) ]
    im = concat_tile(im_tiles)
    return im

def evaluateBlending(images):
    l = len(images)
    r = int(math.sqrt(l))
    shape = np.shape(images[0])
    im = np.zeros(shape, dtype=float)
    for i in range(l):
        im = im + images[i]
    im = im/l
    im = np.array(im, dtype='uint8')
    return im

for fname in fnames:

    images = [cv2.imread(f"{PATH}/{folder_ref}/{subfolder_ref}/prediction_results/{fname}")]

    for elem in others:
        folder, model, subfolder = reference

        fnames_ = [ f for f in os.listdir(f"{PATH}/{folder_ref}/{subfolder_ref}/prediction_results") if f.startswith(fname.replace(".jpg", "").replace(".png", "")) ] 
        stacked_image_meta = (fnames_, f"{PATH}/{folder_ref}/{subfolder_ref}/prediction_results")
        image = getStackedImage(stacked_image_meta)

        images.append(image)
    
    blended_image = evaluateBlending(images)

    cv2.imwrite(f"{PATH}/{EXPERIMENT_ID}/{fname}")