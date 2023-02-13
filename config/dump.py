import yaml
import argparse
import os
import time
import torch
#from torch.utils.tensorboard import SummaryWriter
from  utils import parse_cfg, parse_dump


save_path = "ske_fall_data"

config_Name = "config_v1.yaml" # 参数文件名称
data_ver = 'data_v1'
model_ver = 'model_v1'

log_dir = './runs/{}'.format(time.strftime('%Y.%m.%d-%H-%M',
                            time.localtime(time.time())))    
config={

    'path': {
            'final_model_path': '{}/final'.format(log_dir),
            'best_model_path': '{}/best'.format(log_dir),
            'train_data': "", # 
            'config_path': '{}/config.json'.format(log_dir),
            'save_path': save_path,
            'train_log_path':'{}/TrainLog'.format(log_dir)# tensorboard 数据加载路径
        },

        #  构建模型参数
    'make_model_parameter': {
            'd_input': 16,
            'd_model': 64,
            'in_width': 800,
            'out_width': 400,
            'N': 2
        },

    'batch_size': 64,

    'num_workers': 4,

        # 学习率设置
    'learning_rate': {
            'base_learning_rate': 4.e-5,
            'step_size': 10,
            'gamma': 0.6
        },

        # 训练参数
    'train_epoch': 100,
    'demo_interval': 10,
    'loss_alpha': 0.12,

    # 断点续训练功能
    'load_pretrained_model': False,
    'pretrained_model_path': './runs/{}/final.pt'\
                            .format('2022.12.12-23-01'),

    'train_marks': {
        # 'name': 'mark',
    },
    #最好模型
    'best_model': {
        'epoch': 0,
        'loss': 1000
    }
}

parser = argparse.ArgumentParser()

parser.add_argument("--config_file","-cg",default= os.path.join(os.path.dirname(os.path.realpath(__file__)),config_Name),type=str,help="the config file")
parser.add_argument("--path","-p",default=config['path'],help="")
parser.add_argument('--lr','-lr',default=config['learning_rate'],help='learing rate dict')
parser.add_argument('--epoch','-ep',default=config['train_epoch'],type=int)
parser.add_argument('--interval','-inter',default=config['demo_interval'],type=int)
parser.add_argument('--loss_alpha','-la',default=config['loss_alpha'],type=float)
parser.add_argument('--load_pretraine_model','-preM',default=config['load_pretrained_model'],type=bool,help="断点续训练功能")
parser.add_argument('--pretrained_model_path','-preM_path',default=config['pretrained_model_path'],type=str,help="断点续训练功能")
parser.add_argument('--train_marks','-tm',default=config['train_marks'],help="")
parser.add_argument('--best_model','-bm',default=config[ 'best_model'],help="")
parser.add_argument('--data_version','-dv',default=data_ver,type=str,help="")
parser.add_argument('--model_version','-mv',default=model_ver,type=str,help="")
parser.add_argument('--num_works','-nw',default=config['num_workers'],type=int,help="")
parser.add_argument('--batch_size','-bs',default=config['batch_size'],type=int,help="")


cfg = parser.parse_args()



parse_dump(cfg.config_file,cfg)


if __name__ == '__main__':
    pwd = os.getcwd()
    print(pwd)
    test = parse_cfg(cfg.config_file)
    # print(test.lr['gamma'])
    # print(type(test.lr['gamma']))
