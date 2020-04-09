# Processing images by Doing The Simplest Thing
from pathlib import Path
from PIL import Image
from PIL import ImageFilter
import sys
import concurrent.futures
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

os.chdir('/Users/pratikeshsingh/Desktop/Masters/BigData/Parallel')
in_path = Path.cwd() / "images"
out_path = Path.cwd() / "processed"

def process_one_file(filename):
    outfile = out_path / filename.name
    image = plt.imread(filename)
    gray = rgb2gray(image)
    plt.imsave(outfile,gray,format="png")
    
names = list(in_path.glob("*")) #* 4

import time
start = time.time()
executor = concurrent.futures.ThreadPoolExecutor(99)
list(executor.map(process_one_file, names))
print(time.time() - start)
