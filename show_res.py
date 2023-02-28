import cv2
import yaml
from dataset.dataLoader import create_dataloaders
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

with open('./config/sconfig_v1.yaml') as f:
    config = yaml.safe_load(f)

train_dataset, train_data_loader, val_data_loader = create_dataloaders(
    data_path='data_res/{}'.format(config['dataset']['save_path']),
    batch_size=config['train']['batch_size'],
    num_workers=config['train']['num_workers']
)


def normailize(data):
    _range = np.max(data) - np.min(data)
    return (data-np.min(data))/_range*255


def log_normailize(data):
    max_val = np.log(np.max(data))
    return (np.log(data))/(max_val+1)*255


def show(data):
    pic = log_normailize(data)
    cv2.imwrite('temp.jpg', pic)
    pic = cv2.imread('temp.jpg')
    # cv2.convertScaleAbs(pic,alpha=15)
    im_color = cv2.applyColorMap(pic, cv2.COLORMAP_JET)
    return im_color
    # cv2.imwrite(name,im_color)


def add_mask(pic, tag):
    mask = np.zeros((pic.shape), dtype=np.uint8)
    for j in range(8):
        if tag[0, j, 2] > tag[0, j, 5]:
            index = 2
        else:
            index = 5
        if (tag[0, j, index] > config['train']['threld']):
            mid = int(j*8+tag[0, j, index-2].cpu().detach()*8)
            shift = int(np.floor(tag[0, j, index-1].cpu().detach()*8))
            if (tag[0, j, 6] > tag[0, j, 7]):
                one_mask = cv2.rectangle(mask, (0, np.max([0, mid-shift])), (pic.shape[1], np.min(
                    [pic.shape[0], mid+shift])), color=(0, 255, 0), thickness=-1)
            else:
                one_mask = cv2.rectangle(mask, (0, np.max([0, mid-shift])), (pic.shape[1], np.min(
                    [pic.shape[0], mid+shift])), color=(255, 0, 0), thickness=-1)
            mask += one_mask
    pic_res = cv2.addWeighted(pic, 0.4, mask, 0.6, 0)
    return pic_res


def visualize_sample(rd, ra, re, pic_path, target, predict):
    pic_img = cv2.imread(pic_path)

    plt.subplot(241)
    plt.imshow(pic_img)
    plt.subplot(242)
    rd_img = show(rd[2].detach().cpu().numpy())
    plt.imshow(add_mask(rd_img, target))
    plt.subplot(243)
    rd_img = show(rd[3].detach().cpu().numpy())
    plt.imshow(add_mask(rd_img, target))
    plt.subplot(244)
    rd_img = show(rd[4].detach().cpu().numpy())
    plt.imshow(add_mask(rd_img, target))
    plt.subplot(245)
    rd_img = show(rd[5].detach().cpu().numpy())
    plt.imshow(add_mask(rd_img, predict))
    plt.subplot(246)
    rd_img = show(rd[6].detach().cpu().numpy())
    plt.imshow(add_mask(rd_img, predict))
    plt.subplot(247)
    rd_img = show(rd[7].detach().cpu().numpy())
    plt.imshow(add_mask(rd_img, predict))
    plt.subplot(248)
    rd_img = show(rd[8].detach().cpu().numpy())
    plt.imshow(add_mask(rd_img, predict))
    fig = plt.gcf()
    plt.savefig('res.jpg')
    return fig


def visualize_fall(rd, ra, re, pic_path, target, predict):
    rd_img = show(rd.detach().cpu().numpy())
    ra_img = show(ra.detach().cpu().numpy())
    re_img = show(re.detach().cpu().numpy())
    pic_img = cv2.imread(pic_path)

    # predict[:,:,[0,2,3,5]] = F.sigmoid(predict[:,:,[0,2,3,5]])
    # predict[:,:,[1,4]] = F.relu(predict[:,:,[1,4]])
    # predict[:,:,[6,7]] = F.softmax(predict[:,:,[6,7]])

    plt.subplot(241)
    plt.imshow(pic_img)
    plt.subplot(242)
    plt.imshow(add_mask(rd_img, target))
    plt.subplot(243)
    plt.imshow(add_mask(ra_img, target))
    plt.subplot(244)
    plt.imshow(add_mask(re_img, target))
    plt.subplot(245)
    plt.imshow(add_mask(rd_img, predict))
    plt.subplot(246)
    plt.imshow(add_mask(ra_img, predict))
    plt.subplot(247)
    plt.imshow(add_mask(re_img, predict))
    fig = plt.gcf()
    plt.savefig('res.jpg')
    return fig


if __name__ == '__main__':
    for i, (rd, ra, re, tag, pic) in enumerate(train_data_loader):
        visualize_fall(rd[0][5], ra[0][5], re[0][5], pic[0], tag, tag)
        pass
