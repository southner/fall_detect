from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

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

class MyLoss(nn.Module):
    def __init__(self):
        #loc_loss 本来有预测有的定位loss
        #contain_loss 两个预测框 相应的那个的置信度loss 让它更贴近1
        #not_contain_loss 两个预测框 不相应的那个的置信度loss 让它更贴近0
        #nooobj_loss 本来无预测有的置信度loss
        #class_loss 类别loss
        #total_loss = (l_coord*loc_loss + contain_loss + not_contain_loss + l_noobj*nooobj_loss + class_loss)
        super(MyLoss,self).__init__()
        self.loss = nn.MSELoss()
        self.l_coord = 1
        self.l_noobj = 1
        self.cotain_scale = 1
        self.class_scale = 1
        self.threld = 0.2
    def compute_iou(self,box1, box2):
        pass
        #N M分别为两个box的数目
        N = box1.size(0)
        M = box2.size(0)
        #左上角选择大的
        #距离近点选择大的
        lt = torch.max(
                box1[:,:1].unsqueeze(1).expand(N,M,1),  # [N,1] -> [N,1,1] -> [N,M,1]
                box2[:,:1].unsqueeze(0).expand(N,M,1),  # [M,1] -> [1,M,1] -> [N,M,1]
            )
        #右下角选择小的
        #距离远点选择小的
        rb = torch.min(
                box1[:,1:].unsqueeze(1).expand(N,M,1),  # [N,1] -> [N,1,1] -> [N,M,1]
                box2[:,1:].unsqueeze(0).expand(N,M,1),  # [M,1] -> [1,M,1] -> [N,M,1]
            )
        wh = rb - lt
        wh= torch.clamp(wh,min=0.0)
        inter = wh[:,:,0]
        
        area1 = (box1[:,1]-box1[:,0])  # [N,]
        area2 = (box2[:,1]-box2[:,0])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou #(N,M) (2,1)
    
    def forward(self,pred_tensor,target_tensor):
        # （batch_size,13,4）
        coo_mask = target_tensor[:,:,2] > self.threld   #本来有的mask ==1
        noo_mask = target_tensor[:,:,2] <= self.threld  #本来没有的mask  ==0
        
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)
        
        coo_pred = pred_tensor[coo_mask].view(-1,8) #本来有的对应的预测
        box_pred = coo_pred[:,:6].contiguous().view(-1,3)
        class_pred = coo_pred[:,6:]

        coo_target = target_tensor[coo_mask].view(-1,8)    #本来有的对应的真实值
        box_target = coo_target[:,:6].contiguous().view(-1,3)
        class_target = coo_target[:,6:]

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
        noo_pred = pred_tensor[noo_mask].view(-1,8) #本来没有的对应的预测
        noo_target = target_tensor[noo_mask].view(-1,8)    #本来没有的对应的真实值
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size()) #mask
        noo_pred_mask.zero_()   #初始化全为0
        noo_pred_mask[:,2] = 1
        noo_pred_mask[:,5] = 1
        noo_pred_c = noo_pred[noo_pred_mask]
        noo_target_c = noo_target[noo_pred_mask] 
        nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c,size_average=False)

        coo_response_mask = torch.cuda.ByteTensor(box_target.size()) #[batch_size*13*2,3]
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()

        # 选择最好的IOU 2个box选1个吧 
        for i in range(0,box_target.size()[0],2):   #2是有每个距离bin有两个预测框，i是他们的起始index，之后根据iou选择mask
            # 预测框 2个
            box1 = box_pred[i:i+2]   #box_pred是预测框 shape为(一个batch中target存在的bin数目*2,3)
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            #box1[:,1:2] = abs(box1[:,1:2]) #预测的range可能为负数
            box1_xyxy[:,:1] = box1[:,:1] -0.5*box1[:,1:2]# 下面
            box1_xyxy[:,1:2] = box1[:,:1] +0.5*box1[:,1:2]# 上面
            #  标注框 1个
            box2 = box_target[i].view(-1,3) #box_target是目标框 shape为(一个batch中target存在的bin数目,3)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:,:1] = box2[:,:1] -0.5*box2[:,1:2]
            box2_xyxy[:,1:2] = box2[:,:1] +0.5*box2[:,1:2]

            iou = self.compute_iou(box1_xyxy[:,:2],box2_xyxy[:,:2]) #[2,1]
            max_iou,max_index = iou.max(0)
            max_index = max_index.data.cuda()
            coo_response_mask[i+max_index]=1 # 最大iou对应的mask 值为1 否则为0
            coo_not_response_mask[i+1-max_index]=1# 非最大iou对应的mask 值为1 否则为0

        #本来有预测有 计算是否存在loss 定位loss
        box_pred_response = box_pred[coo_response_mask].view(-1,3)
        box_target_response = box_target[coo_response_mask].view(-1,3)
        contain_loss = F.mse_loss(box_pred_response[:,2],box_target_response[:,2],size_average=False) 
        #temp_loc_loss = F.mse_loss(torch.sqrt(box_pred_response[:,1]),torch.sqrt(box_target_response[:,1]),size_average=False)
        loc_loss = F.mse_loss(box_pred_response[:,0],box_target_response[:,0],size_average=False) 

        #本来有预测无 计算confidence_loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1,3)
        box_target_not_response = box_target[coo_not_response_mask].view(-1,3)
        box_target_not_response[:,2]= 0
        #存在可信度计算   loss的目的是让box_pred_not_response越小越好。就是想让不存在的可能性越小越好
        not_contain_loss = F.mse_loss(box_pred_not_response[:,2],box_target_not_response[:,2],size_average=False)+1e-6

        class_loss = F.mse_loss(class_pred,class_target,size_average=False)

        #loc_loss 本来有预测有的定位loss
        #contain_loss 两个预测框 相应的那个的置信度loss 让它更贴近1
        #not_contain_loss 两个预测框 不相应的那个的置信度loss 让它更贴近0
        #nooobj_loss 本来无预测有的置信度loss
        #class_loss 类别loss
        total_loss = (self.l_coord*loc_loss + class_loss*contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + self.class_scale*class_loss)+1e-6

        return total_loss

if __name__ == '__main__':
    for i in range(10):
        pred = torch.rand((32,13,8),requires_grad=True).cuda()
        target = torch.tensor(pred,requires_grad=True).cuda()
        loss = MyLoss()
        res = loss(pred,target)
        res.backward()
        pass