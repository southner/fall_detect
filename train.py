
from dataset.dataLoader import create_dataloaders
from model.ResAttentionNet import make_model
# from model.AttentionResNet import make_model
import torch
import json
from utils.utils import CRFLoss,MyLoss
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import time
import argparse
import yaml

parser = argparse.ArgumentParser()            # 创建参数解析器
parser.add_argument('--config', default='./config/sconfig_v1.yaml', type=str)   # 添加参数
args = parser.parse_args()  # 解析添加的参数   这个参数是在命令行中输入的

with open(args.config) as f:
        config = yaml.safe_load(f)

log_dir = config['train']['log_path']+'{}'.format(time.strftime('%Y.%m.%d-%H-%M',
                            time.localtime(time.time())))
log_path = {
    'config_path': '{}/config.json'.format(log_dir),
    'final_model_path': '{}/final'.format(log_dir),
    'best_model_path': '{}/best'.format(log_dir),
}
device = torch.device('cuda:0' if torch.cuda.is_available()
                    else 'cpu')
writer = SummaryWriter(log_dir)


def main():
    model = make_model().to(device)
    torch.autograd.set_detect_anomaly(True)
    # 加载已训练模型并添加标注
    if config['train']['load_pretrained_model']:
        model.load_state_dict(
            torch.load(config['train']['pretrained_model_path'],
                       map_location=device)
        )
        writer.add_text('pretrained_model_path',
                        config['train']['pretrained_model_path'])
    else:
        writer.add_text('pretrained_model_path',
                        'NONE')

    # 添加训练标注
    # for name, mark in config['train_marks']:
    #     writer.add_text('train_mark_{}'.format(name), mark)

    train_dataset, train_data_loader, val_data_loader = create_dataloaders(
        data_path='data_res/{}'.format(config['dataset']['save_path']),
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers']
    )

    #criterion = CRFLoss(alpha=config['loss_alpha'])
    criterion = MyLoss()
    val_loss = MyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['train']['learning_rate']['base_learning_rate']
    )
    lr_scheduler = StepLR(
        optimizer,
        step_size=config['train']['learning_rate']['step_size'],
        gamma=config['train']['learning_rate']['gamma']
    )

    epoch = config['train']['epoch']
    
    with open(log_path['config_path'], 'w') as file:
        json.dump(config, file)
        
    for e in range(epoch):

        model.train()   
        train(e, model, train_data_loader, criterion, optimizer)

        model.eval()
        evaluate(e, model, val_data_loader, criterion, optimizer)

        lr_scheduler.step()
        # writer.flush()

        torch.save(model.state_dict(), log_path['final_model_path'])
        
    with open(log_path['config_path'], 'w') as file:
        json.dump(config, file)
        
    writer.close()

    


def train(epoch, model, data_loader, criterion, optimizer):
    total_loss = 0
    total_batch = 0
    for i, (rd,ra,re,tag) in enumerate(data_loader):
        
        rd = rd.to(device)
        ra = ra.to(device)
        re = re.to(device)
        tag = tag.to(device)

        predict= model(rd,ra,re)
        loss = criterion(predict, tag)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batch += 1

        if i % 10 == 0:
            print('epoch:{:0>3d} || batch {:0>3d} with loss:{:>10f} '
                  .format(epoch, i,  loss))

    del rd
    del ra
    del re
    del tag
    del predict
    del loss
    writer.add_scalar('Loss/train', total_loss / total_batch, epoch+1)
    

def evaluate(epoch, model, data_loader, criterion, optimizer):
    # evaluate loss
    total_loss = 0
    total_batch = 0
    for i, (rd,ra,re,tag) in enumerate(data_loader):
        
        rd = rd.to(device)
        ra = ra.to(device)
        re = re.to(device)
        tag = tag.to(device)
        predict= model(rd,ra,re)
        loss = criterion(predict, tag)

        total_loss += loss.item()
        total_batch += 1

    # 保存最佳模型
    average_loss = total_loss / total_batch
    if average_loss < config['train']['best_model']['loss']:
        config['train']['best_model']['loss'] = average_loss
        config['train']['best_model']['epoch'] = epoch
        torch.save(model.state_dict(), log_path['best_model_path'])

    writer.add_scalar('Loss/evaluate', average_loss, epoch)

    # 学习率曲线
    writer.add_scalar('Learning Rate',
                      optimizer.param_groups[0]['lr'],
                      epoch)

    # 效果图
    # if epoch % config['demo_interval'] == 0:
    #     demo_figs = demo(model, device, data_loader, writer,
    #                      epoch, config['path']['data_path'], wave_width=400)
    #     # writer.add_figure('demo/evaluate', demo_figs, epoch, True)
    # pass


if __name__ == '__main__':
    main()
