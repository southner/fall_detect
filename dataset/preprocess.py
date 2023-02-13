from RadDataset import RadDateset
from pathlib import Path
import time
import numpy as np
import json
#将dataset处理为文件 保存下来
import yaml

with open('./config/sconfig_v1.yaml') as f:
        config = yaml.safe_load(f)

def cache_Rad(data_path=config['dataset']['init_data_path'],
            save_path ='data_res/{}'.format(config['dataset']['save_path'])):
    dataset = RadDateset(data_path)
    rd_folder = Path(save_path) / 'cached/rd'
    ra_folder = Path(save_path) / 'cached/ra'
    re_folder = Path(save_path) / 'cached/re'
    tag_folder = Path(save_path) / 'cached/tag'
    if not rd_folder.exists():
        rd_folder.mkdir(parents=True)
    if not ra_folder.exists():
        ra_folder.mkdir(parents=True)
    if not re_folder.exists():
        re_folder.mkdir(parents=True)
    if not tag_folder.exists():
        tag_folder.mkdir(parents=True)
        
    samples_info = []
    sample_num = len(dataset)
    print('========== Preprocessing ==========')
    print('(Preprocessing): {:0>6d} samples in total'.format(sample_num))
    time_start = time.time()

    for i, sample in enumerate(dataset):
        # 以 episode 的起始 sample token 为名
        rd_file = rd_folder / (str(i)+'.npy')
        ra_file = ra_folder / (str(i)+'.npy')
        re_file = re_folder / (str(i)+'.npy')
        tag_file = tag_folder / (str(i)+'.npy')
        # image_file  # 打算在可视化时用

        samples_info.append({
            'rd_file' : str(rd_file),
            'ra_file' : str(ra_file),
            're_file' : str(re_file),
            'tag_file' : str(tag_file),
            # 'image_file': image_file,
        })

        np.save(str(rd_file), sample['RD'],allow_pickle=True)
        np.save(str(ra_file), sample['RA'],allow_pickle=True)
        np.save(str(re_file), sample['RE'],allow_pickle=True)
        np.save(str(tag_file), sample['Tag'],allow_pickle=True)
        
        if i % 100 == 0:
            print('(Preprocessing): {:0>6d} Done || {:0>6d} Left || {:0>5.0f} s'.format(
                i, sample_num-i, time.time()-time_start))

    with open('{}/cached/1.json'.format(save_path), 'w') as file:
        json.dump(samples_info, file)


if __name__ == "__main__":
    cache_Rad()