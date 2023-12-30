import matplotlib.pyplot as plt
import numpy as np
import warnings
import math
import os
import sys
import pandas as pd
from PIL import Image
import random
######## get stress-strain curve from MD simulation result ######
######## find Yield strength, yield strain, UTS and ultimate strain from MD simulation result ######
def stress_strain(df,x):
    yeild = False
    for i in range(5,401):
        df.iloc[i,0] = i*0.0005
        if yeild == False:
            if df.iloc[i,2]>= df.iloc[i+1,2] and df.iloc[i,2]>= df.iloc[i+2,2] and df.iloc[i,2]>= df.iloc[i+5,2] and df.iloc[i,2]>= df.iloc[i-1,2] and df.iloc[i,2]>= df.iloc[i-2,2] and df.iloc[i,2]>= df.iloc[i-5,2]:
                print(i)
                yeild_strain = i*0.0005
                yeild_stress = df.iloc[i,2]
                print ('yeild strain',i*0.0005)
                print ('yeild stress',df.iloc[i,2])
                yeild = True
    UTS = max(df.iloc[:,2])
    UTS_strain = np.argmax(np.array(df.iloc[:,2]))*0.0005
    print ('UTS',UTS)
    print ('UTS_strain',UTS_strain)
    print('xxxxxxx')
    # plt.rcParams.update({'font.size': 14,'font.weight': 'bold'})
    plt.plot(df.iloc[:,0],df.iloc[:,2])
    plt.xlabel('Strain',fontsize=14,fontweight='bold')
    plt.ylabel('Stress/GPa',fontsize=14,fontweight='bold')
    plt.tight_layout()
    plt.savefig(r'/N/project/polycrystalGAN/MNIST_data/Cu_Zr/large_model/no_inverse/ten_stress_strain_curve_image/'+str(x+1)+'.jpg',dpi=600)
    plt.show()
    plt.close()
    return (yeild_strain,yeild_stress,UTS,UTS_strain)
is_train = False
number_of_samples = 100
start_number = 8001
PATH = r'/N/project/polycrystalGAN/MNIST_data/Cu_Zr/large_model/no_inverse/'
file_name = r'/tension-strain-stress.txt'
yeild_stress = np.zeros(number_of_samples)
yeild_strain = np.zeros(number_of_samples)
UTS = np.zeros(number_of_samples)
UTS_strain = np.zeros(number_of_samples)
for i in range(number_of_samples):
    print (i)
    df = pd.DataFrame(pd.read_table(PATH+str(i+start_number)+file_name,sep=' ',header=0))
    [yeild_strain[i],yeild_stress[i],UTS[i],UTS_strain[i]]=stress_strain(df,i)
if is_train == False:
    np.savetxt('ten_yeild_strain_test5.csv', yeild_strain, delimiter=",")
    np.save('ten_yeild_strain_test5.npy',yeild_strain)
    np.savetxt('ten_yeild_stress_test5.csv', yeild_stress, delimiter=",")
    np.save('ten_yeild_stress_test5.npy',yeild_stress)
    np.savetxt(PATH+'ten_UTS_test5.csv', UTS, delimiter=",")
    np.save('ten_UTS_test5.npy',UTS)
    np.savetxt(PATH+'ten_UTS_strain_test5.csv', UTS_strain, delimiter=",")
    np.save('ten_UTS_strain_test5.npy',UTS_strain)
else:
    np.savetxt('ten_yeild_strain.csv', yeild_strain, delimiter=",")
    np.save('ten_yeild_strain.npy',yeild_strain)
    np.savetxt('ten_yeild_stress.csv', yeild_stress, delimiter=",")
    np.save('ten_yeild_stress.npy',yeild_stress)
    np.savetxt(PATH+'ten_UTS.csv', UTS, delimiter=",")
    np.save('ten_UTS.npy',UTS)
    np.savetxt(PATH+'ten_UTS_strain.csv', UTS_strain, delimiter=",")
    np.save('ten_UTS_strain.npy',UTS_strain)
