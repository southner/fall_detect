import scipy.io as scio
import os.path 
import numpy as np

save_dir = r'C:\Users\ZTTTX\Desktop\workspace\experiments\data_save\2022-11-18-16-25-37'
save_path_rd =os.path.join(save_dir,'three_tensor.mat')

data = scio.loadmat(save_path_rd)
print(type(data))
rd = np.array(data['doppler_res'])
ra = np.array(data['azimuth_res'])
re = np.array(data['elevation_res'])
print(rd.shape)