from utils.wsi import WSI
from models.condGAN256 import condGAN256
from helper import *
import argparse
import girder_client

import time

SIZE = 256
LAMDA = 200
LR = 2e-4
OPTIMIZER = 'Adam'

parser = argparse.ArgumentParser(description='Naglah Deep Learning - Model Testing CLI')
parser.add_argument("--model", type= str, default= './model.h5', help="Model File")
parser.add_argument("--inputslide", type= str, default= './input.svs', help="Input WSI")
parser.add_argument("--outputslide", type= str, default= './output.svs', help="Output WSI")
parser.add_argument("--xx", type= float, default= 0, help="Output WSI")
parser.add_argument("--yy", type= float, default= 0, help="Output WSI")
parser.add_argument("--ww", type= str, default= 0, help="Output WSI")
parser.add_argument("--hh", type= str, default= 0, help="Output WSI")


loaded_model = condGAN256()
loaded_model.compile(optimizer=OPTIMIZER, lamda=LAMDA, learning_rate=LR)
g = loaded_model.generator

params = parser.parse_args()
print(params)
MODEL = params.model
INPUTWSI = params.inputslide
OUTPUTWSI = params.outputslide
x = int(params.xx)
y = int(params.yy)
w = int(params.ww)
h = int(params.hh)

g.load_weights(filepath=f'{MODEL}')

svspath = INPUTWSI
print(svspath) 
s = WSI(svspath)
dz = s.GetDeepZoomObject()

print(dz.level_count)
print(dz.tile_count)
print(dz.level_tiles)
print(dz.level_dimensions)

def patchify(im):

    imgheight=im.shape[0]
    imgwidth=im.shape[1]
    size = 256

    cnt_h = imgheight//size+1
    cnt_w = imgwidth//size+1

    patches = []
    for y in range(cnt_h):
        for x in range(cnt_w):
            try:
                im_ = im[x*size:(x+1)*size, y*size:(y+1)*size, :]
            except:
                im__ = np.zeros((size, size, 3), dtype='uint8') + 255
                im_ = im[x:, y:, :]
                a, b, _ = np.shape(im_)
                im__[:a, :b, :] = im_
            patches.append()
    return patches, (cnt_w, cnt_h)

def depatchify(patches, tilemap):
    (cnt_w, cnt_h) = tilemap
    size = 256
    im__ = np.zeros((cnt_w*size, cnt_h*size, 3), dtype='uint8')
    for y in range(cnt_h):
        for x in range(cnt_w):
                im__[x*size:(x+1)*size, y*size:(y+1)*size, :] = patches[y+x*y]
    return im__

def processROI():
    global s, g, OUTPUTWSI, x, y, w, h
    # Save do stain transformation and OUTPUTWSI svs
    im = s.read_patch((x,y), dim=(w,h))
    patches, tilemap = patchify(im)
    im_array = []
    for patch in patches:
        imtf = AdaptBeforePredict(patch)
        im_ = g(imtf, training=True)
        im_ = TF2CV(im_)
        im_array.append(im_)
    im_virtual = depatchify(im_array, tilemap)
    cv2.imwrite(OUTPUTWSI, im_virtual)

    api = "https://athena.rc.ufl.edu/api/v1"
    user = "ahmednaglah"
    password = "Netzwork_171819"

    gc = girder_client.GirderClient(apiUrl=api)
    gc.authenticate(user, password)
    print("\nAuthentication Done!!!!!!!!!!\n")
    gc.upload(OUTPUTWSI, "63ebfd83dd96c4f3a1e539f9", "folder", False, False)
    return True

def processPatch(pnt):
    global g 
    im = s.read_patch(pnt)
    imtf = AdaptBeforePredict(im)
    im_ = g(imtf, training=True)
    im_ = TF2CV(im_)
    return im, im_

status = processROI()
print("\nPrint Status End of Script Again\n")

print("\nPrint Status End of Script\n")
print(status)