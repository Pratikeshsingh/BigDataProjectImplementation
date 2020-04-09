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
import concurrent.futures
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from skimage.color import rgb2gray

#os.chdir('C:/Users/prati/OneDrive/Desktop/Masters/bigdata/code2/Parallel_Image_Processing/images')
os.chdir('/Users/pratikeshsingh/Desktop/Masters/BigData/Parallel')
ROOT_DIR = os.path.abspath("../")

#PATH_TO_C0_TRAIN = os.path.join('C:/Users/prati/OneDrive/Desktop/Masters/bigdata/code2/Parallel_Image_Processing/images','*.JPG')
PATH_TO_C0_TRAIN = os.path.join('/Users/pratikeshsingh/Desktop/Masters/BigData/Parallel/images','*')

TRAIN_FILES = glob.glob(PATH_TO_C0_TRAIN) 
   
N_PROCS = 4

def load_image(img_file):
    return cv2.imread(img_file)

def load_images_parallel(img_files):
    return Parallel(n_jobs=N_PROCS)(delayed(load_image)(img_file) 
                                    for img_file in img_files)
print("_____________________")
start = time.time()
print("Loading train images in parallel")
images_parallel = load_images_parallel(TRAIN_FILES)
print("time taken",time.time() - start, "seconds")
print("images loaded",len(TRAIN_FILES))

out_path = Path.cwd() / "processed"

def process_one_file(filename):
    outfile = out_path / filename
    try:
        image = plt.imread(filename)
        gray = rgb2gray(image)
        plt.imsave(outfile,gray,format="png")
    except IOError:
        print(f"Cannot convert grayscale for {filename}")
print("_____________________")
# names = list(in_path.glob("*")) #* 4
print("Pre Processing images in parallel")
start = time.time()
executor = concurrent.futures.ThreadPoolExecutor(99)
list(executor.map(process_one_file, TRAIN_FILES))
print("time taken",time.time() - start, "seconds")

