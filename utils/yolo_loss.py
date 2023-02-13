import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class Myloss(nn.Module):
    def __init__(self):
        super(Myloss,self).__init__()
        
    def compute_iou(self, box1, box2):
        '''
        计算两个框的重叠率IOU
        通过两组框的联合计算交集，每个框为[x1，y1，x2，y2]。
        Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)
 
        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )
 
        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )
 
        wh = rb - lt  # [N,M,2]
        # wh(wh<0)= 0  # clip at 0
        wh= (wh < 0).float()
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]
 
        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]
 
        iou = inter / (area1 + area2 - inter)
        return iou
    
    def forward(self,pred_tensor,target_tensor):
            '''
            pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
            target_tensor: (tensor) size(batchsize,S,S,30)
            Mr.Li个人见解：
            本来有，预测无--》计算response loss响应损失
            本来有，预测有--》计算not response loss 未响应损失
            本来无，预测无--》无损失(不计算)
            本来无，预测有--》计算不包含obj损失  只计算第4,9位的有无物体概率的loss
            '''
        # 1 找出标注值存在的下标coo_mask与不存在的下标noo_mask
            # N为batchsize
            N = pred_tensor.size()[0]
            # 坐标mask    4：是物体或者背景的confidence    >0   ===========================拿到有物体的记录
            coo_mask = target_tensor[:,:,:,4] > 0
            # 没有物体mask                                 ==0  ===========================拿到无物体的记录
            noo_mask = target_tensor[:,:,:,4] == 0
            # unsqueeze(-1) 扩展最后一维，用0填充，使得形状与target_tensor一样
            # coo_mask、noo_mask形状扩充到[32,7,7,30]
            # coo_mask 大部分为0   记录为1代表真实有物体的网格
            # noo_mask  大部分为1  记录为1代表真实无物体的网格   noo_mask的维度变为target_tensor一样的了  内容用coo_mask填充了 
            coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
            noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)
            # coo_pred 取出预测结果中有物体的网格，并改变形状为（xxx,30）  xxx代表一个batch的图片上的存在物体的网格总数    
            # 30代表2*5+20   例如：coo_pred[72,30]
    
        # 2这段是根据coo_mask与noo_mask在特征图上提取预测框的对应特征值
            coo_pred = pred_tensor[coo_mask].view(-1,30)
            # 一个网格预测的两个box  30的前10即为2个x,y,w,h,c，并调整为（xxx,5） xxx为所有存在标注目标的网格的预测框，形如box_pred[144,5]
            # contiguous将不连续的数组调整为连续的数组
            box_pred = coo_pred[:,:10].contiguous().view(-1,5) #box[x1,y1,w1,h1,c1]
                                                                # #[x2,y2,w2,h2,c2]
            # 每个网格预测的类别  后20
            class_pred = coo_pred[:,10:]
        # 3这段是根据coo_mask在标注特征图上提取对应的网格的特征值
            # 对真实标签做同样操作
            coo_target = target_tensor[coo_mask].view(-1,30)
            box_target = coo_target[:,:10].contiguous().view(-1,5)
            class_target = coo_target[:,10:]
        # 4.下面就是开始具体构建loss函数了
    # # # # # # # # # # # # # # # # 本来无 预测有的损失# # # # # # # # # # # # # # # #  对应橙框
            # 计算不包含obj损失  即本来无，预测有 
            # 在预测结果中拿到真实无物体的网格，并改变形状为（xxx,30）  xxx代表一个batch的图片上的不存在物体的网格总数    30代表2*5+20   例如：[1496,30]
            # 根据noo_mask给出的0 1 信息来提取对应网格  一条数据代表一个网格的信息  noo_mask为 1 说明此网格真实无物体
            noo_pred = pred_tensor[noo_mask].view(-1,30)  
            #提取出标签图像上真实无物体的网格标签内容
            noo_target = target_tensor[noo_mask].view(-1,30)      # 例如：[1496,30]
            # ByteTensor：8-bit integer (unsigned)
            noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size())   # 例如：[1496,30]
            noo_pred_mask.zero_()   #初始化全为0
            # 将第4、9  即将无obj的confidence置为1
            noo_pred_mask[:, 4] = 1
            noo_pred_mask[:, 9] = 1
            # 拿到第4列和第9列里面的值（即拿到真实无物体的网格中，网络预测这些网格有物体的概率值） 一行有两个值（第4和第9位） 
            # 例如noo_pred_c：2992  noo_target_c：2992   如果有obj存在就不会取这个值
            noo_pred_c = noo_pred[noo_pred_mask]
            # 拿到第4列和第9列里面的值  真值为0，表示真实无obj（即拿到真实无物体的网格中，这些网格有物体的概率值，为0）
            noo_target_c = noo_target[noo_pred_mask]
            # 均方误差    如果 size_average = True，返回 loss.mean()。    例如noo_pred_c：2992        noo_target_c：2992
            # nooobj_loss 一个标量  那么这个损失函数目标就是让noo_pred_c无限接近于真值0 这个loss就能无限接近最小值0 损失函数目的达到
            # 想让存在的可能性越小越好
            nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c,size_average=False)
    
    # # # # # # # # # # # # # # # # 本来有 预测有的损失# # # # # # # # # # # # # # # # 
            #计算包含obj损失  即本来有，预测有  和  本来有，预测无
            coo_response_mask = torch.cuda.ByteTensor(box_target.size())
            coo_response_mask.zero_()
            coo_not_response_mask = torch.cuda.ByteTensor(box_target.size())
            coo_not_response_mask.zero_()
            # 选择最好的IOU 2个box选1个吧 
            for i in range(0,box_target.size()[0],2):
                # 预测框 2个
                box1 = box_pred[i:i+2]
                box1_xyxy = Variable(torch.FloatTensor(box1.size()))
                box1_xyxy[:,:2] = box1[:,:2] -0.5*box1[:,2:4]# 左上角
                box1_xyxy[:,2:4] = box1[:,:2] +0.5*box1[:,2:4]# 右下角
                #  标注框 1个
                box2 = box_target[i].view(-1,5)
                box2_xyxy = Variable(torch.FloatTensor(box2.size()))
                box2_xyxy[:,:2] = box2[:,:2] -0.5*box2[:,2:4]
                box2_xyxy[:,2:4] = box2[:,:2] +0.5*box2[:,2:4]
                iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
                max_iou,max_index = iou.max(0)
                max_index = max_index.data.cuda()
                coo_response_mask[i+max_index]=1 # 最大iou对应的mask 值为1 否则为0
                coo_not_response_mask[i+1-max_index]=1# 非最大iou对应的mask 值为1 否则为0
                
            # 1.response loss响应损失，即本来有，预测有   有相应 坐标预测的loss  （x,y,w开方，h开方）参考论文loss公式
            # box_pred [144,5]   coo_response_mask[144,5]   box_pred_response:[72,5]
            # 选择IOU最好的box来进行调整  负责检测出某物体
            box_pred_response = box_pred[coo_response_mask].view(-1,5)# 最佳box坐标提出来其对应的预测值
            box_target_response = box_target[coo_response_mask].view(-1,5)# 最佳box坐标剔除来其对应的真值
            # box_pred_response:[72,5]     计算预测 有物体的概率误差，返回一个数  
            # 存在可信度计算  box_target_response[:,4]的值为1  想让box_pred_response[:,4]存在的可能性越大越好
            contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)   
            # 计算（x,y,w开方，h开方）参考论文loss公式
            # 坐标可信度计算 
            loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False) + F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False)
    
    # # # # # # # # # # # # # # # # 本来有 预测无的损失# # # # # # # # # # # # # # # # 
            # 2.not response loss 未响应损失，即本来有，预测无   未响应
            box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)
            box_target_not_response = box_target[coo_not_response_mask].view(-1,5)
            box_target_not_response[:,4]= 0
            #存在可信度计算   loss的目的是让box_pred_not_response越小越好。就是想让不存在的可能性越小越好
            not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
    
    # # # # # # # # # # # # # # # # 有物体的分类损失# # # # # # # # # # # # # # # # 
            # 3.class loss  计算传入的真实有物体的网格  分类的类别损失 
            class_loss = F.mse_loss(class_pred,class_target,size_average=False)
    # # # # # # # # # # # # # # # # 最终的总损失# # # # # # # # # # # # # # # #         
            # 除以N  即平均一张图的总损失
            return (loc_loss + contain_loss + not_contain_loss + nooobj_loss + class_loss)/N    
        
pred = torch.rand((32,8,8,30),requires_grad=True).cuda()
target = torch.rand((32,8,8,30),requires_grad=True).cuda()
loss = Myloss()
res = loss(pred,target)
res.backward()
