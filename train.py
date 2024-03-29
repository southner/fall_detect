
from dataset.dataLoader import create_dataloaders
from model.ResAttentionNet import make_model
# from model.AttentionResNet import make_model
import torch
import json
from utils.utils import CRFLoss,MyLoss,FallCount
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import time
import argparse
import yaml
from show_res import visualize_fall

import warnings
warnings.filterwarnings("ignore")

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
device = torch.device('cuda:'+config['train']['device'] if torch.cuda.is_available()
                    else 'cpu')
writer = SummaryWriter(log_dir)

# 不显示科学计数法
torch.set_printoptions(sci_mode=False)

def main():
    model = make_model().to(device)
    torch.autograd.set_detect_anomaly(True)
    # 加载已训练模型并添加标注
    if config['train']['load_pretrained_model']:
        model.load_state_dict(
            torch.load("./runs/"+config['train']['pretrained_model_path']+"/best",
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
    count_criterion = FallCount()
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
        train(e, model, train_data_loader, criterion, optimizer,count_criterion)

        model.eval()
        evaluate(e, model, val_data_loader, criterion, optimizer ,count_criterion)

        lr_scheduler.step()
        writer.flush()

        torch.save(model.state_dict(), log_path['final_model_path'])
        
        with open(log_path['config_path'], 'w') as file:
            json.dump(config, file)
        
    writer.close()


def train(epoch, model, data_loader, criterion, optimizer, count_criterion):
    total_loss = 0
    total_sep_fall = torch.tensor([0,0,0,0,0])
    total_sep_heat = torch.tensor([0,0,0])
    
    total_batch = 0
    figure_index = len(data_loader)//10*epoch
    if (config['train']['is_count']):
        #detection classification
        #DTT,DTF,DFT,DFF,Dtotal, CTT,CTF,CFT,CFF,Ctotal
        count = torch.zeros([10])
    for i, ( rd,ra,re,tag,heatmap,pic) in enumerate(data_loader):
        
        rd = rd.to(device)
        ra = ra.to(device)
        tag = tag.to(device)    
        heatmap = heatmap.to(device)

        fall_pred, heat_pred= model(rd,ra)
        loss,sep_fall,sep_heat = criterion(fall_pred, tag, heat_pred, heatmap)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_sep_fall = total_sep_fall + torch.tensor([a.item() for a in sep_fall])
        total_sep_heat = total_sep_heat + torch.tensor([a.item() for a in sep_heat])

        total_batch += 1

        if (config['train']['is_count']):
            count += count_criterion(fall_pred, tag)

        if i % 20 == 0:
            
            print('train epoch:{:0>3d} || batch{:0>3d}'.format(epoch, i))
            print('static : loss {:0>5f} \nloc_loss {:0>5f}  contain_loss {:0>5f} not_contain_loss {:0>5f} nooobj_loss {:0>5f} class_loss {:0>5f} \nbg_loss {:0>5f} jhm_loss {:0>5f} paf_loss {:0>5f}'.\
                      format(total_loss / total_batch, total_sep_fall[0] / total_batch, total_sep_fall[1] / total_batch, total_sep_fall[2] / total_batch, total_sep_fall[3] / total_batch,\
                              total_sep_fall[4] / total_batch, total_sep_heat[0] / total_batch, total_sep_heat[1] / total_batch, total_sep_heat[2] / total_batch))
            if (config['train']['is_count']):
                print('count : {}\n---------------'.format(count))
                # fig = visualize_fall(rd[0,5],ra[0,5],re[0,5],pic[0],tag,predict)
            # writer.add_figure('train_show',fig,figure_index)
            # figure_index+=1
    del rd
    del ra
    del tag
    del heatmap
    del fall_pred
    del heat_pred
    del loss
    writer.add_scalar('train/loss', total_loss / total_batch, epoch+1)
    writer.add_scalar('train/loc_loss', total_sep_fall[0] / total_batch, epoch+1)
    writer.add_scalar('train/contain_loss', total_sep_fall[1] / total_batch, epoch+1)
    writer.add_scalar('train/not_contain_loss', total_sep_fall[2] / total_batch, epoch+1)
    writer.add_scalar('train/nooobj_loss', total_sep_fall[3] / total_batch, epoch+1)
    writer.add_scalar('train/class_loss', total_sep_fall[4] / total_batch, epoch+1)
    writer.add_scalar('train/bg_loss', total_sep_heat[0] / total_batch, epoch+1)
    writer.add_scalar('train/jhm_loss', total_sep_heat[1] / total_batch, epoch+1)
    writer.add_scalar('train/paf_loss', total_sep_heat[2] / total_batch, epoch+1)
    

def evaluate(epoch, model, data_loader, criterion, optimizer , count_criterion):
    # evaluate loss
    total_loss = 0
    total_batch = 0
    total_sep_fall = torch.tensor([0,0,0,0,0])
    total_sep_heat = torch.tensor([0,0,0])
    if (config['train']['is_count']):
        #detection classification
        #DTT,DTF,DFT,DFF,Dtotal, CTT,CTF,CFT,CFF,Ctotal
        count = torch.zeros([10])
    with torch.no_grad():
        for i, ( rd,ra,re,tag,heatmap,pic) in enumerate(data_loader):
            
            rd = rd.to(device)
            ra = ra.to(device)
            tag = tag.to(device)    
            heatmap = heatmap.to(device)

            fall_pred, heat_pred= model(rd,ra)
            loss,sep_fall,sep_heat = criterion(fall_pred, tag, heat_pred, heatmap)

            if (config['train']['is_count']):
                count += count_criterion(fall_pred, tag)
                
            total_loss += loss.item()
            total_sep_fall = total_sep_fall + torch.tensor([a.item() for a in sep_fall])
            total_sep_heat = total_sep_heat + torch.tensor([a.item() for a in sep_heat])

            total_batch += 1

            if i % 50 == 0:
                print('val epoch:{:0>3d} || batch{:0>3d}'.format(epoch, i))
                print('static : loss {:0>5f} \nloc_loss {:0>5f}  contain_loss {:0>5f} not_contain_loss {:0>5f} nooobj_loss {:0>5f} class_loss {:0>5f} \nbg_loss {:0>5f} jhm_loss {:0>5f} paf_loss {:0>5f}'.\
                        format(total_loss / total_batch, total_sep_fall[0] / total_batch, total_sep_fall[1] / total_batch, total_sep_fall[2] / total_batch, total_sep_fall[3] / total_batch,\
                                total_sep_fall[4] / total_batch, total_sep_heat[0] / total_batch, total_sep_heat[1] / total_batch, total_sep_heat[2] / total_batch))
                if (config['train']['is_count']):
                    print('count : {}\n---------------'.format(count))
                
    # 保存最佳模型
    average_loss = total_loss / total_batch
    if average_loss < config['train']['best_model']['loss']:
        config['train']['best_model']['loss'] = average_loss
        config['train']['best_model']['epoch'] = epoch
        torch.save(model.state_dict(), log_path['best_model_path'])

    

    writer.add_scalar('evaluate/loss', total_loss / total_batch, epoch+1)
    writer.add_scalar('evaluate/loc_loss', total_sep_fall[0] / total_batch, epoch+1)
    writer.add_scalar('evaluate/contain_loss', total_sep_fall[1] / total_batch, epoch+1)
    writer.add_scalar('evaluate/not_contain_loss', total_sep_fall[2] / total_batch, epoch+1)
    writer.add_scalar('evaluate/nooobj_loss', total_sep_fall[3] / total_batch, epoch+1)
    writer.add_scalar('evaluate/class_loss', total_sep_fall[4] / total_batch, epoch+1)
    writer.add_scalar('evaluate/bg_loss', total_sep_heat[0] / total_batch, epoch+1)
    writer.add_scalar('evaluate/jhm_loss', total_sep_heat[1] / total_batch, epoch+1)
    writer.add_scalar('evaluate/paf_loss', total_sep_heat[2] / total_batch, epoch+1)
    writer.add_scalar('Loss/detection_precision', count[0]/(count[0]+count[2]), epoch)
    writer.add_scalar('Loss/detection_recall', count[0]/(count[0]+count[1]), epoch)
    writer.add_scalar('Loss/classify_precision', count[5]/(count[5]+count[7]), epoch)
    writer.add_scalar('Loss/classify_recall', count[5]/(count[5]+count[6]), epoch)
    writer.add_text('count',"DTT:{} DTF:{} DFT:{} DFF:{} Dtotal:{} CTT:{} CTF:{} CFT:{} CFF:{} Ctotal:{} ".format(
                    count[0],count[1],count[2],count[3],count[4],
                    count[5],count[6],count[7],count[8],count[9]),epoch)
    # 学习率曲线
    writer.add_scalar('Learning Rate',
                      optimizer.param_groups[0]['lr'],
                      epoch)
    print('val epoch:{:0>3d} || batch{:0>3d} loss{:0>10f}'.format(epoch, i,loss.item()))
    print(count)
    del rd
    del ra
    del tag
    del heatmap
    del fall_pred
    del heat_pred
    del loss
    # 效果图
    # if epoch % config['demo_interval'] == 0:
    #     demo_figs = demo(model, device, data_loader, writer,
    #                      epoch, config['path']['data_path'], wave_width=400)
    #     # writer.add_figure('demo/evaluate', demo_figs, epoch, True)
    # pass

if __name__ == '__main__':
    main()
