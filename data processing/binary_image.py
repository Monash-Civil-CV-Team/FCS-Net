import cv2
import numpy as np
import os, glob
from os import listdir, makedirs
from os.path import isfile, join

path = 'D:/CrackSegNet-master/binary/'  # Source Folder
dstpath = 'D:/CrackSegNet-master/new_data/test/background/mask/'  # Destination Folder
try:
    makedirs(dstpath)
except:
    print("Directory already exist, images will be written in same folder")
# Folder won't used
files = list(filter(lambda f: isfile(join(path, f)), listdir(path)))
for image in files:
    try:
        img = cv2.imread(os.path.join(path, image))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = np.ones_like(gray)
        inv[np.where(gray > 1)] = 0
        bi = inv * 255
        dstPath = join(dstpath, image)

        cv2.imwrite(dstPath, bi)

    except:
        print("{} is not converted".format(image))

