from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import yaml

with open('./config/sconfig_v1.yaml') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda:'+config['train']['device'] if torch.cuda.is_available()
                    else 'cpu')

class SeparateLoss(nn.Module):
    def __init__(self):
        super(SeparateLoss, self).__init__()

    def forward(self, predict, target):
        return torch.mean(0.25-(predict-0.5)**2)

class CRFLoss(nn.Module):
    """
    smooth_l1 + alpha * separate_loss
    """

    def __init__(self, alpha=0.5):
        super(CRFLoss, self).__init__()
        self.alpha = alpha
        self.smooth_l1 = nn.SmoothL1Loss()
        
    def forward(self, ske_predict, target):
        return self.smooth_l1(ske_predict, target)

class FallLoss(nn.Module):
    def __init__(self):
        #loc_loss 本来有预测有的定位loss
        #contain_loss 两个预测框 相应的那个的置信度loss 让它更贴近1
        #not_contain_loss 两个预测框 不相应的那个的置信度loss 让它更贴近0
        #nooobj_loss 本来无预测有的置信度loss
        #class_loss 类别loss
        #total_loss = (l_coord*loc_loss + contain_loss + not_contain_loss + l_noobj*nooobj_loss + class_loss)
        super(FallLoss,self).__init__()
        self.loss = nn.MSELoss()
        self.cross_entropy = nn.BCELoss()
        self.loc_scale = 1.4
        self.cotain_scale = 4
        self.not_cotain_scale = 3
        self.l_noobj = 1.2
        self.class_scale = 0.5
        # self.fall_vs_norm_scale = 10
        self.threld = config['train']['threld']
    def compute_iou(self,box1, box2):
        '''
        计算两个框的重叠率IOU
        通过两组框的联合计算交集，每个框为[range1,amuith1,range2,amuith2]。
        Args:
          box1: (tensor) pred bounding boxes, sized [N,4].
          box2: (tensor) target bounding boxes, sized [M,4].
          N=2 M=1
        Return:
          (tensor) iou, sized [N,M].
        '''
        #N M分别为两个box的数目
        N = box1.size(0)
        M = box2.size(0)
        #左上角选择大的
        #距离近点选择大的
        lt = torch.max(
                box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,1] -> [N,1,1] -> [N,M,1]
                box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,1] -> [1,M,1] -> [N,M,1]
            )
        #右下角选择小的
        #距离远点选择小的
        rb = torch.min(
                box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,1] -> [N,1,1] -> [N,M,1]
                box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,1] -> [1,M,1] -> [N,M,1]
            )
        wh = rb - lt
        wh= torch.clamp(wh,min=0.0)
        inter = wh[:,:,0] * wh[:,:,1]
        
        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou #(N,M) (2,1)
    
    def forward(self,pred_tensor,target_tensor):
        # （batch_size,8,9,12）
        # (batch_size,range,amuith,box*2+label)
        # box (range_mid,range, amuith_mid,amuith,confidence) label (nofall,fall)
        # pred_tensor[:,:,[0,2,3,5]] = F.sigmoid(pred_tensor[:,:,[0,2,3,5]])
        # pred_tensor[:,:,[1,4]] = F.relu(pred_tensor[:,:,[1,4]])
        # pred_tensor[:,:,[6,7]] = F.softmax(pred_tensor[:,:,[6,7]])
        
        coo_mask = target_tensor[:,:,:,4] > self.threld   #本来有的mask ==1
        noo_mask = target_tensor[:,:,:,4] <= self.threld  #本来没有的mask  ==0

        # coo_fall_mask = (target_tensor[:,:,2] > self.threld) & (target_tensor[:,:,7]==1)
        # coo_normal_mask = (target_tensor[:,:,2] > self.threld) & (target_tensor[:,:,6]==1)

        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)
        # coo_fall_mask = coo_fall_mask.unsqueeze(-1).expand_as(target_tensor)
        # coo_normal_mask = coo_normal_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_pred = pred_tensor[coo_mask].view(-1,12) #本来有的对应的预测
        box_pred = coo_pred[:,:10].contiguous().view(-1,5)
        class_pred = coo_pred[:,10:]
        
        coo_target = target_tensor[coo_mask].view(-1,12)    #本来有的对应的真实值
        box_target = coo_target[:,:10].contiguous().view(-1,5)
        class_target = coo_target[:,10:]

        

        #loss分为3部分 
        # obj_loss 是否存在目标
        #   本来无预测无 无obj_loss
        #   本来无预测有 计算obj_loss 为confidence
        #   本来有预测有 无obj_loss 计算w_loss和label_loss
        #   本来有预测无 选择框 计算obj_loss 为confidence
        # w_loss 目标框loss，在我们的项目中只有距离
        # label_loss 类别loss

        # 计算不包含obj损失  即本来无，预测有 
        # 取出本来没有的mask 将pred与target（全为0）计算loss
        noo_pred = pred_tensor[noo_mask].view(-1,12) #本来没有的对应的预测
        noo_target = target_tensor[noo_mask].view(-1,12)    #本来没有的对应的真实值
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size(),device=device) #mask
        noo_pred_mask.zero_()   #初始化全为0
        noo_pred_mask[:,4] = 1  #取出两个box的confidence
        noo_pred_mask[:,9] = 1
        noo_pred_c = noo_pred[noo_pred_mask]
        noo_target_c = noo_target[noo_pred_mask] 
        # 一个缩小loss的小trick，预测confidence<threld-0.3 损失记为0 size_average不确定
        nooobj_loss = F.mse_loss(F.relu(noo_pred_c-self.threld+0.3),noo_target_c,size_average=False)

        coo_response_mask = torch.cuda.ByteTensor(box_target.size(),device=device) #[batch_size*13*2,3]
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size(),device=device)
        coo_not_response_mask.zero_()

        # 选择最好的IOU 2个box选1个吧 
        for i in range(0,box_target.size()[0],2):   #box_target.size() [target_num*2,5]
            # 预测框 2个
            box1 = box_pred[i:i+2]   #box_pred是预测框 [target_num*2,5]
            box1_xyxy = Variable(torch.FloatTensor(box1.size())) # shape[2,5]
            #box1[:,1:2] = abs(box1[:,1:2]) #预测的range可能为负数
            box1_xyxy[:,:2] = box1[:,:2] -0.5*box1[:,2:4]# 左上角
            box1_xyxy[:,2:4] = box1[:,:2] +0.5*box1[:,2:4]# 右下角
            #  标注框 1个
            box2 = box_target[i].view(-1,5) #box_target是目标框
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:,:2] = box2[:,:2] -0.5*box2[:,2:4]
            box2_xyxy[:,2:4] = box2[:,:2] +0.5*box2[:,2:4]
            
            iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
            max_iou,max_index = iou.max(0)
            max_index = max_index.data.cuda(device=device)
            coo_response_mask[i+max_index]=1 # 最大iou对应的mask 值为1 否则为0
            coo_not_response_mask[i+1-max_index]=1# 非最大iou对应的mask 值为1 否则为0

        #本来有预测有 计算是否存在loss 定位loss
        box_pred_response = box_pred[coo_response_mask].view(-1,5)
        box_target_response = box_target[coo_response_mask].view(-1,5)
        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False) 
        #temp_loc_loss = F.mse_loss(torch.sqrt(box_pred_response[:,1]),torch.sqrt(box_target_response[:,1]),size_average=False)
        loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False) + F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False)

        #本来有预测无 计算confidence_loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1,5)
        box_target_not_response[:,4]= 0
        #存在可信度计算   loss的目的是让box_pred_not_response越小越好。就是想让不存在的可能性越小越好
        not_contain_loss = F.mse_loss(box_pred_not_response[:,4],box_target_not_response[:,4],size_average=False)

        # cross_entropy loss
        # class_target_fall = target_tensor[coo_fall_mask].view(-1,8)[:,7]
        # class_target_normal = target_tensor[coo_normal_mask].view(-1,8)[:,6]
        # class_pred_fall = pred_tensor[coo_fall_mask].view(-1,8)[:,7]
        # class_pred_normal = pred_tensor[coo_normal_mask].view(-1,8)[:,6]
        # class_fall_loss = self.cross_entropy(class_pred_fall,class_target_fall) if len(class_pred_fall)>0 else 0
        # class_normal_loss = self.cross_entropy(class_pred_normal+1e-0006,class_target_normal)
        # class_loss =  self.fall_vs_norm_scale*class_fall_loss + class_normal_loss


        class_loss = F.mse_loss(class_pred,class_target,size_average=False)

        #loc_loss 本来有预测有的定位loss
        #contain_loss 两个预测框 相应的那个的置信度loss 让它更贴近1
        #not_contain_loss 两个预测框 不相应的那个的置信度loss 让它更贴近0
        #nooobj_loss 本来无预测有的置信度loss
        #class_loss 类别loss
        total_loss = (self.loc_scale*loc_loss + self.cotain_scale*contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + self.class_scale*class_loss)
        # total_loss = class_loss
        return total_loss,torch.tensor([self.loc_scale*loc_loss,self.cotain_scale*contain_loss,not_contain_loss,self.l_noobj*nooobj_loss,self.class_scale*class_loss])
        
class FallCount(nn.Module):
    def __init__(self):
        #loc_loss 本来有预测有的定位loss
        #contain_loss 两个预测框 相应的那个的置信度loss 让它更贴近1
        #not_contain_loss 两个预测框 不相应的那个的置信度loss 让它更贴近0
        #nooobj_loss 本来无预测有的置信度loss
        #class_loss 类别loss
        #total_loss = (l_coord*loc_loss + contain_loss + not_contain_loss + l_noobj*nooobj_loss + class_loss)
        super(FallCount,self).__init__()
        self.threld = config['train']['threld']
    def forward(self,pred_tensor,target_tensor):
        #计算 detection 
        #计算 classify
        # pred_tensor[:,:,[0,2,3,5]] = F.sigmoid(pred_tensor[:,:,[0,2,3,5]])
        # pred_tensor[:,:,[1,4]] = F.relu(pred_tensor[:,:,[1,4]])
        # pred_tensor[:,:,[6,7]] = F.softmax(pred_tensor[:,:,[6,7]])
        
        
        #计算detection指标 detect_correct：batch_size*bin_num中检测对了多少
        #target 中是否存在目标
        detect_tar_mask = target_tensor[:,:,:,4] > self.threld   
        #选择confidence较大的box
        detect_pred_choose = torch.max(pred_tensor[:,:,:,4],pred_tensor[:,:,:,9])
        #pred 中是否存在目标
        detect_pred_mask = detect_pred_choose > self.threld  

        DTT = torch.sum((detect_tar_mask==1) & (detect_pred_mask==1),dim=[0,1,2]) 
        DTF = torch.sum((detect_tar_mask==1) & (detect_pred_mask==0),dim=[0,1,2]) 
        DFT = torch.sum((detect_tar_mask==0) & (detect_pred_mask==1),dim=[0,1,2]) 
        DFF = torch.sum((detect_tar_mask==0) & (detect_pred_mask==0),dim=[0,1,2]) 
        Dtotal = DTT + DTF + DFT + DFF

        #计算classfy指标
        #有对象，需要分类的数据mask
        classsify_mask = target_tensor[:,:,:,4] > self.threld   
        classsify_mask = classsify_mask.unsqueeze(-1).expand_as(target_tensor)
        
        #label为倒数两位
        classsify_pred = pred_tensor[classsify_mask].view(-1,12)[:,-2:]
        classsify_targ = target_tensor[classsify_mask].view(-1,12)[:,-2:]

        #第7位是非跌倒 第8位是跌倒 若非跌倒概率高 为1
        classsify_pred = torch.where(classsify_pred[:,0] > classsify_pred[:,1],1,0)
        classsify_targ = torch.where(classsify_targ[:,0] > classsify_targ[:,1],1,0)

        CTT = sum((classsify_targ==1) & (classsify_pred==1))              #跌倒 预测 为跌倒
        CTF = sum((classsify_targ==1) & (classsify_pred==0))              #跌倒 预测为 非跌倒
        CFT = sum((classsify_targ==0) & (classsify_pred==1))              #非跌倒 预测 为跌倒
        CFF = sum((classsify_targ==0) & (classsify_pred==0))              #非跌倒 预测 为非跌倒

        Ctotal = CTT + CTF + CFT + CFF

        #detect时是否有目标判断正确的bin个数 判断错误的bin个数
        #对于已知grountruth、有目标的bin，classify时 跌倒预测为跌倒的个数
        return torch.tensor([DTT,DTF,DFT,DFF,Dtotal,CTT,CTF,CFT,CFF,Ctotal],requires_grad=False)

class HeatLoss(nn.Module):
    def __init__(self):
        super(HeatLoss, self).__init__()
        self.lambda_1 = 2
        self.lambda_2 = 500
        self.lambda_3 = 100
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.k_jhm = 1
        self.b_jhm = 1
        self.k_paf = 1
        self.b_paf = 0.3

    def forward(self, predict, heatmap):
        loss_sm = self.bce_loss(predict[:, 25, :, :], heatmap[:, 25, :, :])
        weight_jhm = self.k_jhm*heatmap[:, 0:25, :, :]+self.b_jhm * \
            torch.where(heatmap[:, 0:25, :, :] >= 0, 1, -1)
        weight_paf = self.k_paf*heatmap[:, 26:, :, :]+self.b_paf * \
            torch.where(heatmap[:, 26:, :, :] >= 0, 1, -1)
        loss_jhm = torch.mean(weight_jhm*self.mse_loss(predict[:, 0:25, :, :],heatmap[:, 0:25, :, :]))
        loss_paf = torch.mean(weight_paf*self.mse_loss(predict[:, 26:, :, :],heatmap[:, 26:, :, :]))
        total = self.lambda_1*loss_sm + self.lambda_2*loss_jhm + self.lambda_3*loss_paf
        return total,torch.tensor([self.lambda_1*loss_sm,self.lambda_2*loss_jhm,self.lambda_3*loss_paf])

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.fall_loss = FallLoss()
        self.heat_loss = HeatLoss()
        self.lambda_1 = 0.2

    def forward(self, fall_predict, fall_target, heat_predict, heat_target):
        fall,sep_fall = self.fall_loss(fall_predict,fall_target)
        heat,sep_heat = self.heat_loss(heat_predict,heat_target)
        total = self.lambda_1*fall+(1-self.lambda_1)*heat
        return total,self.lambda_1*sep_fall,(1-self.lambda_1)*sep_heat
        
if __name__ == '__main__':
    # res = torch.zeros([10])
    lossF  = FallLoss()
    for i in range(10):
        pred = torch.rand((32,8,9,12),requires_grad=True).to(device)
        target = torch.rand((32,8,9,12),requires_grad=True).to(device)
        # pred[:,1:5,:,4] = 1
        # target[:,1:5,:,4] = 1
        # loss,sep_loss = lossF(pred,target)

        pred[:, :, :, [0, 2, 4, 5, 7, 9]] = F.sigmoid(pred[:, :, :, [0, 2, 4, 5, 7, 9]])
        pred[:, :, :, [1, 3, 6, 8]] = F.relu(pred[:, :, :, [1, 3, 6, 8]])
        pred[:, :, :, [10, 11]] = F.softmax(pred[:, :, :, [10, 11]], dim=3)
        
        target[:, :, :, [0, 2, 4, 5, 7, 9]] = F.sigmoid(target[:, :, :, [0, 2, 4, 5, 7, 9]])
        target[:, :, :, [1, 3, 6, 8]] = F.relu(target[:, :, :, [1, 3, 6, 8]])
        target[:, :, :, [10, 11]] = F.softmax(target[:, :, :, [10, 11]], dim=3)
        
        count = FallCount()
        count(pred,target)
        # res += loss(pred,target)
        pass