from pathlib import Path
import numpy as np
import scipy.io as scio
from os import listdir
import os.path as opt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split, SubsetRandomSampler
import torch
import yaml

with open('./config/sconfig_v1.yaml') as f:
        config = yaml.safe_load(f)

class RadDateset(Dataset):
    #将文件处理为sample，交给cache函数存下来
    def __init__(self,
                data_path=config['dataset']['init_data_path'],
                shuffle=False):
        super(RadDateset).__init__()
        self.data_path = data_path

        #参数
        self.window_step = 5
        self.window_size = 10
        experiments_config=pd.read_excel(config['dataset']['experiments_config_path'],keep_default_na=False)
        train_pos = []#非跌倒
        train_neg = []#跌倒
        for i in range(len(experiments_config['dir_name'].values)):
            if experiments_config['person'][i] in config['dataset']['train_config']['person'] and experiments_config['place'][i] in config['dataset']['train_config']['place']:
                if experiments_config['kind'][i]=='跌倒':
                    train_neg.append(experiments_config.values[i])
                else:
                    train_pos.append(experiments_config.values[i])
        
        #处理数据
        self.samples = []
        sample_index = 0 
        for dir_name in train_pos:#非跌倒 label为0
            dir = opt.join(self.data_path,dir_name[0])
            frame_num = len(listdir(opt.join(dir,'color_res')))
            radar_data = scio.loadmat(opt.join(dir,'three_tensor.mat'))
            # 将图片长宽颠倒，变为(range,another)
            radar_data_doppler = np.transpose(radar_data['doppler_res'],(0,-1,-2))
            radar_data_azimuth = np.transpose(radar_data['azimuth_res'],(0,-1,-2))
            radar_data_elevation = np.transpose(radar_data['elevation_res'],(0,-1,-2))
            
            range_data = np.load(opt.join(dir,'skeleton_range_res.npy'))
            for frame_index_begin in range(0,frame_num-30,2): #0-27 2-29 ... 414-441
                slice = [i for i in range(frame_index_begin,frame_index_begin+30,3)]
                sample = {}
                sample['RD'] = radar_data_doppler[slice,1:,:]
                sample['RA'] = radar_data_azimuth[slice,1:,:]
                sample['RE'] = radar_data_elevation[slice,1:,:]
                sample['Tag'] = np.zeros([13,8]) #(mid,range,confidence)*2+label(one_hot)
                mid = frame_index_begin+15
                for person_id in range(0,3):
                    person_range = range_data[mid,person_id]
                    if (np.mean(person_range)==0):
                        break
                    if (np.mean(person_range)>=config['dataset']['range_config']['range_max']):
                        continue
                    range_index = int(np.mean(person_range)/config['dataset']['range_config']['range_res']//1)
                    
                    sample['Tag'][range_index,0] = np.mean(person_range)/config['dataset']['range_config']['range_res']%1
                    sample['Tag'][range_index,1] = (person_range[1]-person_range[0])/2/config['dataset']['range_config']['range_res']
                    sample['Tag'][range_index,2] = 1
                    sample['Tag'][range_index,3] = np.mean(person_range)/config['dataset']['range_config']['range_res']%1
                    sample['Tag'][range_index,4] = (person_range[1]-person_range[0])/2/config['dataset']['range_config']['range_res']
                    sample['Tag'][range_index,5] = 1
                    sample['Tag'][range_index,6] = 1 #非跌倒
                    sample['Tag'][range_index,7] = 0 #跌倒
                    pass
                self.samples.append(sample)
    
        for dir_name in train_neg:#跌倒 label为1
            dir = opt.join(self.data_path,dir_name[0])
            frame_num = len(listdir(opt.join(dir,'color_res')))
            radar_data = scio.loadmat(opt.join(dir,'three_tensor.mat'))
            # 将图片长宽颠倒，变为(range,another)
            radar_data_doppler = np.transpose(radar_data['doppler_res'],(0,-1,-2))
            radar_data_azimuth = np.transpose(radar_data['azimuth_res'],(0,-1,-2))
            radar_data_elevation = np.transpose(radar_data['elevation_res'],(0,-1,-2))
            range_data = np.load(opt.join(dir,'skeleton_range_res.npy'))
            
            for fall_index in range(1,4):
                if dir_name[-fall_index]=='':
                    continue
                for bias in range(-5,8):  #23-53   
                    slice = [i for i in range(fall_index-15+bias,fall_index+15+bias,3)]
                    # 0-39  20  5-32 0-27 12-40
                    sample = {}
                    sample['RD'] = radar_data_doppler[slice,1:,:]
                    sample['RA'] = radar_data_azimuth[slice,1:,:]
                    sample['RE'] = radar_data_elevation[slice,1:,:]
                    sample['Tag'] = np.zeros([13,8]) #(mid,range,confidence,label)
                    mid = frame_index_begin+15
                    for person_id in range(0,1):
                        person_range = range_data[mid,person_id]
                        if (np.mean(person_range)==0):
                            break
                        if (np.mean(person_range)>=config['dataset']['range_config']['range_max']):
                            continue
                        range_index = int(np.mean(person_range)/config['dataset']['range_config']['range_res']//1)
                        
                        sample['Tag'][range_index,0] = np.mean(person_range)/config['dataset']['range_config']['range_res']%1
                        sample['Tag'][range_index,1] = (person_range[1]-person_range[0])/2/config['dataset']['range_config']['range_res']
                        sample['Tag'][range_index,2] = 1
                        sample['Tag'][range_index,3] = np.mean(person_range)/config['dataset']['range_config']['range_res']%1
                        sample['Tag'][range_index,4] = (person_range[1]-person_range[0])/2/config['dataset']['range_config']['range_res']
                        sample['Tag'][range_index,5] = 1
                        sample['Tag'][range_index,6] = 0 #非跌倒
                        sample['Tag'][range_index,7] = 1 #跌倒
                        pass
                    self.samples.append(sample)
            

        pass
    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
        
        

if __name__ =="__main__":
    dataset = RadDateset()
    pass
