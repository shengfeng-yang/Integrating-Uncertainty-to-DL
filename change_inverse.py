import matplotlib.pyplot as plt
import numpy as np
import warnings
import ovito
import math
import os
import sys
import random
from ovito.vis import Viewport
import pandas as pd
warnings.filterwarnings("ignore")
from PIL import Image
import gzip
import numpy as np
###### Prepare the atomic model for MD simulation #####


###### read images from MNIST ######

f = gzip.open('./train-images-idx3-ubyte.gz','r')

image_size = 28
num_images =10001

PATH = os.getcwd()
num =  int("".join(list(filter(str.isdigit, PATH))))

f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)


#### change to binary image ####
import matplotlib.pyplot as plt
image = np.asarray(data[num]).squeeze()
image_bi = np.zeros([28,28])
for i in range(28):
    for j in range(28):
        if image[i,j] > 128:
            image_bi[i,j] = 255
        else:
            image_bi[i,j] = 0
# plt.imshow(image)
# plt.show()
# plt.imshow(image_bi)
# plt.show()
image_bi_a = []
image_bi_b = []
for i in range (28):
    for j in range (28):
        if image_bi[i,j] == 0:
            image_bi_a.append([i,j])
        else:
            image_bi_b.append([i,j])
#### show labels of MNIST ####
f = gzip.open('./train-labels-idx1-ubyte.gz','r')
f.read(8)
for i in range(0,50):   
    buf = f.read(1)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    print(labels)

##### creat MNIST.cfg #####

filename1 = './final1.cfg'
pipeline = ovito.io.import_file(filename1)
data = pipeline.compute()
position = data.particles['Position']
cell = data.cell[:,:]
pixel_length = cell[0,0]/28

position_a = []
for i in range(len(position)):
    position_a.append([position[i,0],position[i,1],position[i,2]])


pos_a_select = []
pos_b_select = []
pos_b_list = []
pos_c_list = []
for i in range(len(position_a)):
    for j in range(len(image_bi_b)):
        if position_a[i][0] <= pixel_length*image_bi_b[j][0]+pixel_length+cell[0,3] and position_a[i][0] >= pixel_length*image_bi_b[j][0]+cell[0,3]:
            if position_a[i][1] <= pixel_length*image_bi_b[j][1]+pixel_length+cell[1,3] and position_a[i][1] >= pixel_length*image_bi_b[j][1]+cell[1,3]:
                if random.random()<0.76:
                    pos_b_list.append(i)
                else:
                    pos_c_list.append(i)
pos_b_list = list(set(pos_b_list))
pos_c_list = list(set(pos_c_list))

########## write down the LAMMPS datafile ########
with open (PATH+'/test_1_new_inverse.lmp','w') as f:
    f.write(' # change the number of atoms with density\n')
    f.write('\n')
    f.write(str(len(position_a)-len(pos_c_list))+' atoms\n')
    f.write(' 2  atom types\n')
    f.write(str(cell[0,3])+'  '+str(cell[0,0]+cell[0,3])+'  xlo  xhi\n')
    f.write(str(cell[1,3])+'  '+str(cell[1,1]+cell[1,3])+'  ylo  yhi\n')
    f.write(str(cell[2,3])+'  '+str(cell[2,2]+cell[2,3])+'  zlo  zhi\n')
    f.write('\n')
    f.write('Masses\n')
    f.write('\n')
    f.write('   1   63.546    # Cu \n')
    f.write('   2   91.224      # Zr \n')
    f.write('\n')
    f.write('Atoms #atomic\n')
    f.write('\n')
    atom_id = 0
    count_c = 0
    count_b = 0
    c = []
    b = []
    for i in range(len(position_a)):
        if i in pos_c_list:
            count_c += 1
            c.append(i)
            continue
        elif i in pos_b_list:
            atom_id += 1
            b.append(i)
            count_b += 1
            if random.random()<0.64:
                f.write(str(atom_id)+'  1  '+ str(position_a[i][0])+' '+str(position_a[i][1])+' '+str(position_a[i][2])+'\n')
            else:
                f.write(str(atom_id)+'  2  '+ str(position_a[i][0])+' '+str(position_a[i][1])+' '+str(position_a[i][2])+'\n') 
        else:
            atom_id += 1
            f.write(str(atom_id)+'  1  '+ str(position_a[i][0])+' '+str(position_a[i][1])+' '+str(position_a[i][2])+'\n')
    f.close
print('finish')