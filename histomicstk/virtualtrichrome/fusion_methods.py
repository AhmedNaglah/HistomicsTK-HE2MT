import cv2
import os
import numpy as np
import pywt

def retreiveImages(gt_fname, root, sizes):
    fname = gt_fname.replace("real", "fake")
    filenames = [ f"{root}/{k}/{fname}" for k in sizes ]

    def readImagesAndTimes(filenames):

        images = []
        for filename in filenames:
            im = cv2.imread(filename)
            images.append(im)

        return images

    images = readImagesAndTimes(filenames)
    return images

#### ExposureFusion
def ExposureFusion(images):

    mergeMertens = cv2.createMergeMertens()
    exposureFusion = mergeMertens.process(images)
    exposureFusion = (exposureFusion - np.min(exposureFusion))/(np.max(exposureFusion)-np.min(exposureFusion))
    exposureFusion = np.array(exposureFusion*255, dtype='uint8')

    return exposureFusion

#### DWT
def DWTNImages(images):

    # This function does the coefficient fusing according to the fusion method
    def fuseCoeff(cooef1, cooef2, method):

        if (method == 'mean'):
            cooef = (cooef1 + cooef2) / 2
        elif (method == 'min'):
            cooef = np.minimum(cooef1,cooef2)
        elif (method == 'max'):
            cooef = np.maximum(cooef1,cooef2)
        else:
            cooef = []

        return cooef

    # Params
    FUSION_METHOD = 'mean' # Can be 'min' || 'max || anything you choose according theory
    
    def DWT(a,b):
        wavelet = 'db1'
        cooef1 = pywt.wavedec2(a[:,:], wavelet)
        cooef2 = pywt.wavedec2(b[:,:], wavelet)

        # Second: for each level in both image do the fusion according to the desire option
        fusedCooef = []
        for i in range(len(cooef1)-1):
            # The first values in each decomposition is the apprximation values of the top level
            if(i == 0):

                fusedCooef.append(fuseCoeff(cooef1[0],cooef2[0],FUSION_METHOD))
            else:
                # For the rest of the levels we have tupels with 3 coeeficents
                c1 = fuseCoeff(cooef1[i][0], cooef2[i][0],FUSION_METHOD)
                c2 = fuseCoeff(cooef1[i][1], cooef2[i][1], FUSION_METHOD)
                c3 = fuseCoeff(cooef1[i][2], cooef2[i][2], FUSION_METHOD)
                fusedCooef.append((c1,c2,c3))

        # Third: After we fused the cooefficent we nned to transfor back to get the image
        fusedImage = pywt.waverec2(fusedCooef, wavelet)

        # Forth: normmalize values to be in uint8
        fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
        fusedImage = fusedImage.astype(np.uint8)
        fusedImage = cv2.resize(fusedImage, (256,256))

        return fusedImage

    def DWT2Images(im1, im2):
        # Read the two image
        I1 = images[0]
        I2 = images[1]

        I1B = I1[:,:,0]
        I1G = I1[:,:,1]
        I1R = I1[:,:,2]

        I2B = I2[:,:,0]
        I2G = I2[:,:,1]
        I2R = I2[:,:,2]

        IB = DWT(I1B, I2B)
        IG = DWT(I1G, I2G)
        IR = DWT(I1R, I2R)

        fusedImage = np.zeros(np.shape(images[0]), dtype='uint8')
        fusedImage[:,:,0] = IB
        fusedImage[:,:,1] = IG
        fusedImage[:,:,2] = IR

        return fusedImage

    out = images[0]
    for i in range(len(images)-1):
        im = images[i+1]
        out = DWT2Images(out, im)

    return out
