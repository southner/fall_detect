#将skeleton_res转换为skeleton_range_res
import os.path as opt
import os
import pandas as pd
import numpy as np
import yaml

with open('./config/sconfig_v1.yaml') as f:
    config = yaml.safe_load(f)

data_path = config['dataset']['init_data_path']

for dir_name in os.listdir(config['dataset']['init_data_path']):
    dir = opt.join(data_path,dir_name)
    range_data = np.load(opt.join(dir,'skeleton_res.npy'))
    range_data_res = np.zeros([450,3,2])
    for i in range(450):
        for j in range(3):
            one_range_data = range_data[i,j]**2
            range_res = np.sqrt(one_range_data[:,0] + one_range_data[:,2])
            range_data_res[i,j,0] = np.percentile(range_res,10)
            range_data_res[i,j,1] = np.percentile(range_res,90)
    pass
    with open(opt.join(dir,'skeleton_range_xz_res.npy'),'wb') as f:
        np.save(f,range_data_res)