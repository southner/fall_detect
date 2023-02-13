from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import torch
from torch.utils.data import random_split, SubsetRandomSampler

class RadDatesetCacheDataset(Dataset):
    def __init__(self,
                 data_path='data',
                 version='1'):

        super(RadDatesetCacheDataset, self).__init__()

        self.data_path = data_path
        
        with open('{}/cached/{}.json'.format(data_path, version), 'r') as file:
            self.samples = json.load(file)

    def __getitem__(self, i):
        rd = np.load(self.samples[i]['rd_file'],allow_pickle=True)
        ra = np.load(self.samples[i]['ra_file'],allow_pickle=True)
        re = np.load(self.samples[i]['re_file'],allow_pickle=True)
        tag = np.load(self.samples[i]['tag_file'],allow_pickle=True)
        return rd[:,:],ra,re,tag

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def collate_fn(batch):
        rd,ra,re,tag  = zip(*batch)
        rd = np.stack(rd).astype(np.float32)
        ra = np.stack(ra).astype(np.float32)
        re = np.stack(re).astype(np.float32)
        tag = np.stack(tag).astype(np.float32)
        return torch.tensor(rd,requires_grad=True),torch.tensor(ra,requires_grad=True),torch.tensor(re,requires_grad=True),torch.tensor(tag,requires_grad=True)
        
def create_dataloader(
    data_path='data',
    version='1',
    batch_size=8,
    num_workers=1,
):

    dataset = RadDatesetCacheDataset(data_path=data_path, version=version)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=RadDatesetCacheDataset.collate_fn,
    )
    return dataset, dataloader


def create_dataloaders(
    data_path='ske_fall_data',
    version='1',
    train_ratio=0.8,
    batch_size=32,
    random_seed=125,
    num_workers=4
):
    """根据指定版本数据集`划分出训练集和验证集`,并分别生成 dataloader
    """
    assert(train_ratio > 0 and train_ratio < 1)

    dataset = RadDatesetCacheDataset(data_path=data_path,
                                    version=version)
    
    sample_count = len(dataset)
    train_index, val_index = \
        random_split(range(sample_count),
                     [round(train_ratio * sample_count),
                      round((1-train_ratio)*sample_count)],
                     generator=torch.Generator().manual_seed(random_seed))
    train_index = [sample for sample in train_index]
    val_index = [sample for sample in val_index]
    np.random.shuffle(train_index)
    np.random.shuffle(val_index)

    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)
    
    train_loader = DataLoader(dataset, batch_size,
                              num_workers=num_workers,
                              sampler=train_sampler,
                              collate_fn=RadDatesetCacheDataset.collate_fn)
    val_loader = DataLoader(dataset, batch_size,
                            num_workers=num_workers,
                            sampler=val_sampler,
                            collate_fn=RadDatesetCacheDataset.collate_fn)

    return dataset, train_loader, val_loader


if __name__ == "__main__" : 
    dataset, train_loader, val_loader = create_dataloaders()
    for i, ( rd,ra,re,tag) in enumerate(train_loader):
        pass
        # aa = Skeleton[0,3:].reshape((-1,32,6)).numpy()
        # source_data = aa[1,:,:3]
        # trans_data = source_data[:]
        # mean = np.mean(source_data,axis=(0))
        # for i in range(3):
        #     trans_data[:,i]=source_data[:,i]-mean[i]
        # point_cloud = open3d.geometry.PointCloud()
        # point_cloud.points = open3d.utility.Vector3dVector(source_data)
        # point_cloud.paint_uniform_color((0,0,255))
        # open3d.visualization.draw_geometries([point_cloud],point_show_normal=True)
        # point_cloud.points = open3d.utility.Vector3dVector(trans_data)
        # point_cloud.paint_uniform_color((0,255,0))
        # open3d.visualization.draw_geometries([point_cloud],point_show_normal=True)
        pass 
    for i, (RD, RA, RE, Skeleton) in enumerate(val_loader):
        pass 