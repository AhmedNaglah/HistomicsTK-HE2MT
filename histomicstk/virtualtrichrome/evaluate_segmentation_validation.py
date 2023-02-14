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


PATH = "F:/N002-Research/liver-pathology/segmentation/"
EXPERIMENT_ID = 'Evaluate_validation_samples'

f1 = open(f'{PATH}/generic.csv', 'w')
header = "dataset,model,fname,ACC,iou,iouA,iouB,dice,DSC"
print(header, file=f1)

f3 = open(f'{PATH}/{EXPERIMENT_ID}_Summary.out', 'w')
f4 = open(f'{PATH}/{EXPERIMENT_ID}_Abstract.out', 'w')

print(f'Experiment Summary: {EXPERIMENT_ID}', file=f3)
print(f'-----------------------------------', file=f3)

cols = ['dataset', 'model', 'fname', 'acc', 'iou', 'iouA', 'iouB', 'dice', 'dice2']
df = pd.DataFrame(columns=cols)

dataset ="validation_samples"
models = os.listdir(f"{PATH}/{dataset}/output")
for model in models:
    # Load Images at directory
    datadir = f"{PATH}/{dataset}/output/{model}"
    fnames = os.listdir(datadir)

    for i in range(len(fnames)):
        fname = fnames[i]
        input_image, gt, prediction = load_image_segmentation(f'{datadir}/{fname}')
        acc, iou, iouA, iouB, dice, dice2 = segmentation_metrics(gt, prediction)
        row = {'dataset': dataset, 'model': model, 'fname':fname, 'acc':acc, 'iou':iou, 'iouA':iouA, 'iouB':iouB, 'dice':dice, 'dice2':dice2}
        df = df.append(row, ignore_index=True)
        if i%200==0:
            print(f"Processing {i}")
        model2 = model.replace(" ","").replace("_", "-").replace("GAN-", "GAN+")
        dataset2 = dataset
        print(f'{dataset2},{model2},{fname},{acc},{iou},{iouA},{iouB},{dice},{dice2}', file=f1)

    df_summary = df.loc[df['model']==model ]
    df_summary = df_summary.loc[df_summary['dataset']==dataset]

    print(f'-----------------------------------', file=f3)
    print(f'----Dataset: {dataset}-------------', file=f3)
    print(f'----Model: {model}-----------------', file=f3)
    print(f'-----------------------------------', file=f3)

    print(df_summary.mean(), file=f3)
    print(df_summary.std(), file=f3)
    print(f'-----------------------------------', file=f3)
    print(df_summary.describe(), file=f3)

    print(f"Done, Experiment ID:{EXPERIMENT_ID}")

    a = df_summary[['acc']].mean(axis=0).values[0]
    b = df_summary[['acc']].std(axis=0).values[0]
    c = df_summary[['dice2']].mean(axis=0).values[0]
    d = df_summary[['dice2']].std(axis=0).values[0]

    #print(f'{dataset};{model};{a},{b};{c},{d}', file=f4)
    print(f"('{dataset}',[('{model}',[({a:.02f},{b:.02f}),({c:.02f},{d:.02f})])]),", file=f4)

    #DATA = [('Hepatic Artery Branch', [('HE2F', [(0.74, 0.06), (0.86, 0.02)])])]

f1.close()

f3.close()