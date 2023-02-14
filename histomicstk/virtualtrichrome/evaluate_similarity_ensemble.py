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
    hc = abs(cv2.compareHist(histBr.astype(dtype=np.float32), histBg.astype(dtype=np.float32), method ))
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

def similarity_metrics(triple):
    _, w, _ = np.shape(triple)
    w = w//3
    im1 = triple[:,w:(2*w),:]
    im2 = triple[:,(2*w):,:]
    mi = getMI(im1, im2)
    nmi = getNMI(im1, im2)
    hc = getHC(im1, im2)
    bcd = getBCD(im1, im2)

    return mi,nmi,hc,bcd

def TF2CV(im):
    img = tf.cast(tf.math.scalar_mul(255/2, im[0]+1), dtype=tf.uint8)
    img_ = np.array(tf.keras.utils.array_to_img(img),dtype='uint8')
    img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
    return img_

def getMT(im_, model):
    im2 = np.expand_dims(im_, axis=0)/255
    out = model.call(im2)
    return TF2CV(out)

def load_triple(image_file):

    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    _, w, _ = np.shape(image)
    w = w // 3

    input_image = image[:, :w, :]
    gt = image[:, w:(2*w), :]
    prediction = image[:, (2*w):, :]

    return cv2.hconcat((input_image, gt, prediction))

def load_image_similarity(image_file):
  triple  = load_triple(image_file)

  return triple

def createDir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False 


""" parser = argparse.ArgumentParser(description='Naglah Segmentation Metrics')
parser.add_argument("--dataroot", required= True, help="root directory that contains the data")
parser.add_argument("--experiment_id", required= True, type=str, help="Experiment ID to track experiment and results" )
parser.add_argument("--model", required= True, type=str, help="Experiment ID to track experiment and results" )
parser.add_argument("--experiment_id_old", required= True, type=str, help="Experiment ID to track experiment and results" )

params = parser.parse_args()

PATH = params.dataroot
EXPERIMENT_ID = params.experiment_id
EXPERIMENT_ID_OLD = params.experiment_id_old
MODEL = params.model """

PATH = "D:/Out_temp"
EXPERIMENT_ID = 'similarity_loo_fold4'
FOLDER = 'fold4' 

f1 = open(f'{PATH}/{EXPERIMENT_ID}.csv', 'w')
header = "dataset,model,fname,mi,nmi,hc,bcd"
print(header, file=f1)

f3 = open(f'{PATH}/{EXPERIMENT_ID}_Summary.out', 'w')
f4 = open(f'{PATH}/{EXPERIMENT_ID}_Abstract.out', 'w')

print(f'Experiment Summary: {EXPERIMENT_ID}', file=f3)
print(f'-----------------------------------', file=f3)

cols = ['folder',  'fname', 'mi', 'nmi', 'hc', 'bcd']
df = pd.DataFrame(columns=cols)

# Load Images at directory
datadir = f"{PATH}/{FOLDER}"
fnames = os.listdir(datadir)
saveDir = f"{PATH}/{EXPERIMENT_ID}"
createDir(saveDir)

for i in range(len(fnames)):
    fname = fnames[i]
    triple = load_image_similarity(f'{datadir}/{fname}')
    #cv2.imwrite(f"{saveDir}/{fname}", triple)
    mi, nmi, hc, bcd = similarity_metrics(triple)
    row = {'folder': FOLDER, 'fname':fname, 'mi':mi, 'nmi':nmi, 'hc':hc, 'bcd':bcd}
    df = df.append(row, ignore_index=True)
    if i%200==0:
        print(f"Processing {i}")
    print(f'{FOLDER},{fname},{mi},{nmi},{hc},{bcd}', file=f1)

df_summary = df.loc[df['folder']==FOLDER ]

print(f'-----------------------------------')
print(f'----Folder: {FOLDER}-------------')
print(f'-----------------------------------')

print(df_summary.mean())
print(df_summary.std())
print(f'-----------------------------------')
print(df_summary.describe())

print(f"Done, Experiment ID:{EXPERIMENT_ID}")

a = df_summary[['mi']].mean(axis=0).values[0]
b = df_summary[['mi']].std(axis=0).values[0]

c = df_summary[['nmi']].mean(axis=0).values[0]
d = df_summary[['nmi']].std(axis=0).values[0]

e = df_summary[['hc']].mean(axis=0).values[0]
f = df_summary[['hc']].std(axis=0).values[0]

g = df_summary[['bcd']].mean(axis=0).values[0]
h = df_summary[['bcd']].std(axis=0).values[0]

print(f"{FOLDER};{a},{b};{c},{d};{e},{f};{g},{h};", file=f3)
print(f"('{FOLDER}',[('{EXPERIMENT_ID}',[({a:.02f},{b:.02f}),({c:.02f},{d:.02f}),({e:.02f},{f:.02f}),({g:.02f},{h:.02f})])]),", file=f4)

#DATA = [('Hepatic Artery Branch', [('HE2F', [(0.74, 0.06), (0.86, 0.02)])])]

f1.close()

f3.close()