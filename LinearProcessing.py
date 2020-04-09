# Processing images by Doing The Simplest Thing
from pathlib import Path
from PIL import Image
from PIL import ImageFilter
import sys
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

os.chdir('/Users/pratikeshsingh/Desktop/Masters/BigData/Parallel')
in_path = Path.cwd() / "images"
out_path = Path.cwd() / "processed"

import time
start = time.time()
names = list(in_path.glob("*"))
for filename in names:
    outfile = out_path / filename.name
    image = plt.imread(filename)
    gray = rgb2gray(image)
    plt.imsave(outfile,gray,format="png")    

print(time.time() - start)
