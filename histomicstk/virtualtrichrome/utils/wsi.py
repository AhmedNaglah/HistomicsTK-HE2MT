import os
#os.add_dll_directory("C:/Users/inagl/Downloads/openslide-win64-20171122/openslide-win64-20171122/bin/")
#os.add_dll_directory("C:/Users/inagl/Desktop/BE544/openslide-win64-20220811/openslide-win64-20220811/bin")

import openslide
import numpy as np
import cv2
from openslide.deepzoom import DeepZoomGenerator

class WSI():

    def __init__(self, filepath):
        self.slide = openslide.open_slide(filepath)

    def WSI2CSV(self, im):
        im = np.array(im.convert("RGB"))
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        return im
    
    def GetDeepZoomObject(self):
        return DeepZoomGenerator(self.slide, tile_size=254, overlap=1, limit_bounds=False)

    def span(self):
        return self.slide.level_downsamples[self.slide.level_count-1]
    
    def d_low(self):
        return self.slide.level_dimensions[self.slide.level_count-1]
    
    def d_high(self):
        return self.slide.level_dimensions[0]

    def level_count(self):
        return self.slide.level_count

    def read_lowres(self):
        try:
            return self.WSI2CSV(self.slide.read_region((0,0), self.slide.level_count-1, self.slide.level_dimensions[self.slide.level_count-1]))
        except:
            return -1

    def read_lowres_pil(self):
        try:
            return self.slide.read_region((0,0), self.slide.level_count-1, self.slide.level_dimensions[self.slide.level_count-1])
        except:
            return -1

    def read_patch(self, corner, dim=(256, 256)):
        try:
            return self.WSI2CSV(self.slide.read_region(corner, 0, dim))
        except:
            return -1      

    def read_patch_pil(self, corner, dim=(256, 256)):
        try:
            return self.slide.read_region(corner, 0, dim)
        except:
            return -1   

    def read_level(self, level):
        try:
            return self.WSI2CSV(self.slide.read_region((0,0), level, self.slide.level_dimensions[level]))
        except:
            return -1

    def read_highres(self):
        try:
            return self.WSI2CSV(self.slide.read_region((0,0), 0, self.slide.level_dimensions[0]))
        except:
            return -1