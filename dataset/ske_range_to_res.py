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
    range_data_res = np.zeros([450,3,4])
    for i in range(450):
        for j in range(3):
            # x y z 右 下 前

            one_range_data = range_data[i,j]
            one_range_data_temp = one_range_data**2
            range_res = np.sqrt(one_range_data_temp[:,0] + one_range_data_temp[:,1] + one_range_data_temp[:,2])
            # arcsin 和 arctan结果一样   arccos 无法区分左右
            angle_res = np.arcsin(one_range_data[:,0]/range_res)/np.pi*180
            # 距离计算
            range_data_res[i,j,0] = np.percentile(range_res,5)
            range_data_res[i,j,1] = np.percentile(range_res,95)
            # 方位角计算
            range_data_res[i,j,2] = np.percentile(angle_res,5)
            range_data_res[i,j,3] = np.percentile(angle_res,95)
            pass
    pass

    with open(opt.join(dir,'skeleton_RandA_res.npy'),'wb') as f:
        np.save(f,range_data_res)