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
import cv2 as cv


with open('./config/sconfig_v1.yaml') as f:
        config = yaml.safe_load(f)

def log_norm(dt):
    dt = np.log(dt+1+1e-6)
    max = np.max(dt)
    dt = dt/max
    return dt


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
        pos_count = 0
        neg_count = 0
        sample_index = 0 
        for dir_name in train_pos:#非跌倒 label为0
            dir = opt.join(self.data_path,dir_name[0])
            frame_num = len(listdir(opt.join(dir,'color_res')))
            radar_data = scio.loadmat(opt.join(dir,'three_tensor.mat'))
            # 将图片长宽颠倒，变为(range,another)
            radar_data_doppler = np.transpose(radar_data['doppler_res'],(0,-1,-2))
            radar_data_azimuth = np.transpose(radar_data['azimuth_res'],(0,-1,-2))
            radar_data_elevation = np.transpose(radar_data['elevation_res'],(0,-1,-2))
            radar_data_doppler = log_norm(radar_data_doppler)
            radar_data_azimuth = log_norm(radar_data_azimuth)
            radar_data_elevation = log_norm(radar_data_elevation)
            
            location_data = np.load(opt.join(dir,'skeleton_RandA_res.npy'))
            range_data = location_data[:,:,:2]/1000
            amuith_data = location_data[:,:,2:]
            # x_data = np.load(opt.join(dir,'skeleton_range_res.npy'))
            for frame_index_begin in range(0,frame_num-30,2): #0-27 2-29 ... 414-441
                slice = [i for i in range(frame_index_begin,frame_index_begin+30,3)]
                sample = {}
                sample['RD'] = radar_data_doppler[slice,1:,:]
                sample['RA'] = radar_data_azimuth[slice,1:,:]
                sample['RE'] = radar_data_elevation[slice,1:,:]
                #(range_mid,amuith_mid,range,amuith,confidence)*2+label(nofall,fall)
                sample['Tag'] = np.zeros([config['dataset']['range_config']['range_num'],config['dataset']['amuith_config']['amuith_num'],12]) 
                sample['pic'] = opt.join(dir,'color_res/{:0>5d}.jpg'.format(frame_index_begin+12))

                # 数据中心帧
                mid = frame_index_begin+15

                # 获取heatmap
                heatmap = cv.imread(opt.join(
                dir, 'open_body/{:0>5d}.png'.format(mid)),0)
                if (heatmap is None):
                    continue
                heatmap = heatmap.reshape([46,78,-1]).transpose([1,0,2])
                #26-77为paf paf大部分值为128 129
                heatmap[26:] = np.abs(129-heatmap[26:])
                sample['heatmap'] = heatmap/255

                # 获取跌倒label
                is_useful = False
                for person_id in range(0,3):
                    person_range = range_data[mid,person_id]
                    person_amuith = amuith_data[mid,person_id]
                    if (np.mean(person_range)==0):
                        break
                    if (np.mean(person_range)>=config['dataset']['range_config']['range_max']):
                        continue
                    range_index = int(np.mean(person_range)/config['dataset']['range_config']['range_res'])
                    amuith_index = int((np.mean(person_amuith)+90)/(180/config['dataset']['amuith_config']['amuith_num']))

                    is_useful = True
                    sample['Tag'][range_index,amuith_index,0] = np.mean(person_range)/config['dataset']['range_config']['range_res']%1 
                    sample['Tag'][range_index,amuith_index,1] = (np.mean(person_amuith)+90)/(180/config['dataset']['amuith_config']['amuith_num'])%1
                    sample['Tag'][range_index,amuith_index,2] = (person_range[1]-person_range[0])/2/config['dataset']['range_config']['range_res']
                    sample['Tag'][range_index,amuith_index,3] = (person_amuith[1]-person_amuith[0])/2/(180/config['dataset']['amuith_config']['amuith_num'])
                    sample['Tag'][range_index,amuith_index,4] = 1

                    sample['Tag'][range_index,amuith_index,5] = np.mean(person_range)/config['dataset']['range_config']['range_res']%1
                    sample['Tag'][range_index,amuith_index,6] = (np.mean(person_amuith)+90)/(180/config['dataset']['amuith_config']['amuith_num'])%1
                    sample['Tag'][range_index,amuith_index,7] = (person_range[1]-person_range[0])/2/config['dataset']['range_config']['range_res']
                    sample['Tag'][range_index,amuith_index,8] = (person_amuith[1]-person_amuith[0])/2/(180/config['dataset']['amuith_config']['amuith_num'])
                    sample['Tag'][range_index,amuith_index,9] = 1
                    
                    sample['Tag'][range_index,amuith_index,10] = 1 #非跌倒
                    sample['Tag'][range_index,amuith_index,11] = 0 #跌倒

                    

                    pass
                if (is_useful):
                    self.samples.append(sample)
                    pos_count+=1
        for dir_name in train_neg:#跌倒 label为1
            dir = opt.join(self.data_path,dir_name[0])
            frame_num = len(listdir(opt.join(dir,'color_res')))
            radar_data = scio.loadmat(opt.join(dir,'three_tensor.mat'))
            # 将图片长宽颠倒，变为(range,another)
            radar_data_doppler = np.transpose(radar_data['doppler_res'],(0,-1,-2))
            radar_data_azimuth = np.transpose(radar_data['azimuth_res'],(0,-1,-2))
            radar_data_elevation = np.transpose(radar_data['elevation_res'],(0,-1,-2))
            radar_data_doppler = log_norm(radar_data_doppler)
            radar_data_azimuth = log_norm(radar_data_azimuth)
            radar_data_elevation = log_norm(radar_data_elevation)
            
            location_data = np.load(opt.join(dir,'skeleton_RandA_res.npy'))
            range_data = location_data[:,:,:2]/1000
            amuith_data = location_data[:,:,2:]

            for fall_person_index in range(1,4):
                if dir_name[-fall_person_index]=='':
                    continue
                fall_index = dir_name[-fall_person_index]
                for bias in range(-5,8):  #23-53   
                    
                    slice = [i for i in range(fall_index-15+bias,fall_index+15+bias,3)]
                    # 0-39  20  5-32 0-27 12-40
                    sample = {}
                    sample['RD'] = radar_data_doppler[slice,1:,:]
                    sample['RA'] = radar_data_azimuth[slice,1:,:]
                    sample['RE'] = radar_data_elevation[slice,1:,:]
                    sample['Tag'] = np.zeros([config['dataset']['range_config']['range_num'],config['dataset']['amuith_config']['amuith_num'],12]) #(range_mid,range,amuith_mid,amuith,confidence)*2+label(one_hot)
                    sample['pic'] = opt.join(dir,'color_res/{:0>5d}.jpg'.format(fall_index+bias))

                    mid = fall_index+bias

                    # 获取heatmap
                    heatmap = cv.imread(opt.join(
                    dir, 'open_body/{:0>5d}.png'.format(mid)),0)
                    if (heatmap is None):
                        continue
                    heatmap = heatmap.reshape([46,78,-1]).transpose([1,0,2])
                    #26-77为paf paf大部分值为128 129
                    heatmap[26:] = np.abs(129-heatmap[26:])
                    sample['heatmap'] = heatmap/255

                    # 获取跌倒label
                    is_useful = False
                    for person_id in range(0,1):
                        person_range = range_data[mid,person_id]
                        person_amuith = amuith_data[mid,person_id]
                        if (np.mean(person_range)==0):
                            continue
                        if (np.mean(person_range)>=config['dataset']['range_config']['range_max']):
                            continue
                        range_index = int(np.mean(person_range)/config['dataset']['range_config']['range_res']//1)
                        is_useful = True
                        sample['Tag'][range_index,amuith_index,0] = np.mean(person_range)/config['dataset']['range_config']['range_res']%1 
                        sample['Tag'][range_index,amuith_index,1] = (np.mean(person_amuith)+90)/(180/config['dataset']['amuith_config']['amuith_num'])%1
                        sample['Tag'][range_index,amuith_index,2] = (person_range[1]-person_range[0])/2/config['dataset']['range_config']['range_res']
                        sample['Tag'][range_index,amuith_index,3] = (person_amuith[1]-person_amuith[0])/2/(180/config['dataset']['amuith_config']['amuith_num'])
                        sample['Tag'][range_index,amuith_index,4] = 1

                        sample['Tag'][range_index,amuith_index,5] = np.mean(person_range)/config['dataset']['range_config']['range_res']%1
                        sample['Tag'][range_index,amuith_index,6] = (np.mean(person_amuith)+90)/(180/config['dataset']['amuith_config']['amuith_num'])%1
                        sample['Tag'][range_index,amuith_index,7] = (person_range[1]-person_range[0])/2/config['dataset']['range_config']['range_res']
                        sample['Tag'][range_index,amuith_index,8] = (person_amuith[1]-person_amuith[0])/2/(180/config['dataset']['amuith_config']['amuith_num'])
                        sample['Tag'][range_index,amuith_index,9] = 1

                        sample['Tag'][range_index,amuith_index,10] = 0 #非跌倒
                        sample['Tag'][range_index,amuith_index,11] = 1 #跌倒
                        pass
                    if (is_useful):
                        self.samples.append(sample)
                        neg_count+=1
        print('pos sample num :{},neg sample num :{}'.format(pos_count,neg_count))

        pass
    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
        
        

if __name__ =="__main__":
    dataset = RadDateset()
    pass
