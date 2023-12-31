import matplotlib.pyplot as plt
import numpy as np
import warnings
import math
import os
import sys
import random
import pandas as pd
warnings.filterwarnings("ignore")
from PIL import Image
import gzip
import numpy as np

# generate the input for the ML model

f = gzip.open('./train-images-idx3-ubyte.gz','r')

image_size = 28
num_images =10001

# PATH = os.getcwd()
# num =  int("".join(list(filter(str.isdigit, PATH))))

f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)


#### 转换为二值图 ####
# import matplotlib.pyplot as plt
is_train = False
num_of_samples = 980
start_number = 9021
image_bi = np.zeros([num_of_samples,28,28])
for num in range(start_number,start_number+num_of_samples):
    image = np.asarray(data[num]).squeeze()
    for i in range(28):
        for j in range(28):
            if image[i,j] > 128:
                image_bi[num-start_number,i,j] = 255
            else:
                image_bi[num-start_number,i,j] = 0
if is_train == False:
    np.save('image_bi_test3.npy',image_bi)
else:
    np.save('image_bi.npy',image_bi)

# plt.imshow(image)
# plt.show()
# plt.imshow(image_bi)
# plt.show()