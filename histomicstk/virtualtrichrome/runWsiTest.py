from utils.wsi import WSI
from models.condGAN256 import condGAN256
from helper import *
import argparse

import time

SIZE = 256
LAMDA = 200
LR = 2e-4
OPTIMIZER = 'Adam'

parser = argparse.ArgumentParser(description='Naglah Deep Learning - Model Testing CLI')
parser.add_argument("--model", type= str, default= './model.h5', help="Model File")
parser.add_argument("--input", type= str, default= './input.svs', help="Input WSI")
parser.add_argument("--output", type= str, default= './output.svs', help="Output WSI")

loaded_model = condGAN256()
loaded_model.compile(optimizer=OPTIMIZER, lamda=LAMDA, learning_rate=LR)
g = loaded_model.generator

try:
    params = parser.parse_args()
    MODEL = params.model
    INPUTWSI = params.input
    OUTPUTWSI = params.output

    g.load_weights(filepath=f'{MODEL}')

    svspath = INPUTWSI
    print(svspath) 
    s = WSI(svspath)
    dz = s.GetDeepZoomObject()

    print(dz.level_count)
    print(dz.tile_count)
    print(dz.level_tiles)
    print(dz.level_dimensions)

    def processAllPatches():
        global s, g, OUTPUTWSI
        # Save do stain transformation and OUTPUTWSI svs
        return True
        
    def processPatch(pnt):
        global g 
        im = s.read_patch(pnt)
        imtf = AdaptBeforePredict(im)
        im_ = g(imtf, training=True)
        im_ = TF2CV(im_)
        return im, im_
    
    status = processAllPatches()
    print("\nPrint Status End of Script\n")
    print(status)

except:
    print('Error Code NAGLAH000 - Script Error - Check Parameters')
