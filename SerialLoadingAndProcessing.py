#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:37:44 2020

@author: pratikeshsingh
"""

import datetime
import time
import cv2
import os
import glob
from joblib import Parallel, delayed
from pathlib import Path
from PIL import Image
from PIL import ImageFilter
import sys
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


os.chdir('/Users/pratikeshsingh/Desktop/Masters/BigData/Parallel')
ROOT_DIR = os.path.abspath("../")
PATH_TO_C0_TRAIN = os.path.join('/Users/pratikeshsingh/Desktop/Masters/BigData/Parallel/images','*')

TRAIN_FILES = glob.glob(PATH_TO_C0_TRAIN)    
N_PROCS = 2


def load_images(img_files):
    imgs = []
    for img_file in img_files:
        imgs.append(load_image(img_file))
    return imgs

def load_image(img_file):
    return cv2.imread(img_file)                     
              
print("_____________________")
start = time.time()
print("Loading images in series")
images = load_images(TRAIN_FILES)
print("time taken",time.time() - start, "seconds")
print("images loaded",len(TRAIN_FILES))

print("_____________________")
out_path = Path.cwd() / "out"
print("Pre Processing images in series")

import time
start = time.time()
for filename in TRAIN_FILES:
    outfile = out_path / filename
    image = plt.imread(outfile)
    gray = rgb2gray(image)
    plt.imsave(outfile,gray,format="png")    

print("time taken",time.time() - start, "seconds")

