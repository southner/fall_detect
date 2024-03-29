from torch import nn
import torch
if __name__=='__main__':
    from utils import clones
else:
    from .utils import clones
import torch.nn.functional as F


class NormConv(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size=1,stride = 1,padding="same"):
        super(NormConv, self).__init__()
        
        self.m = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride,padding=padding),
            # BatchNorm2d是针对通道的归一化，具体对于(N,C,W,H)的输入，针对每一个C，计算一个均值和方差
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), 
        )

    def forward(self, x):
        # x.shape = (-1, features, width)
        # x = self.m(x).squeeze()
        # x.shape = (-1, width)
        return self.m(x)
    
class NormMaxPoll(nn.Module):
    def __init__(self, featuires,kernel_size=1,stride=1,padding=0):
        super(NormMaxPoll, self).__init__()

        self.m = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size,stride=stride,padding=padding),
            nn.BatchNorm2d(featuires),
            nn.ReLU(), 
        )

    def forward(self, x):
        return self.m(x)

class ResProjectionUnit(nn.Module):
    '''
    模型开始前的预卷积
    '''
    def __init__(self, in_channels,temp_channels,out_channels,stride=2):
        super(ResProjectionUnit, self).__init__()

        self.branch1 = nn.Sequential(
            NormConv(in_channels,temp_channels,kernel_size=1,stride=stride,padding="valid"),
            NormConv(temp_channels,temp_channels,kernel_size=3),
            nn.Conv2d(temp_channels,out_channels,kernel_size=1)
        )
        self.branch2 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,padding="valid")


    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        res = b1+b2
        return res

class ResUnit(nn.Module):
    '''
    模拟文章中的ResUnit
    '''
    def __init__(self, in_channels):
        super(ResUnit, self).__init__()
        temp_channel = int(in_channels/4)
        self.BR = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(), 
        )
        self.conv = nn.Sequential(
            NormConv(in_channels,temp_channel,1),
            NormConv(temp_channel,temp_channel,3),
            nn.Conv2d(temp_channel,in_channels,1),
        )

    def forward(self, x):
        b1 = x
        b2 = self.conv(self.BR(x))
        res = b1+b2
        return res

class NormConv1d(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size=1,stride = 1,padding="same"):
        super(NormConv1d, self).__init__()
        
        self.m = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride = stride,padding=padding),
            # BatchNorm2d是针对通道的归一化，具体对于(N,C,W,H)的输入，针对每一个C，计算一个均值和方差
            nn.BatchNorm1d(out_channels),
            nn.ReLU(), 
        )

    def forward(self, x):
        # x.shape = (-1, features, width)
        # x = self.m(x).squeeze()
        # x.shape = (-1, width)
        return self.m(x)

class Res1dUnit(nn.Module):
    '''
    模拟文章中的ResUnit
    '''
    def __init__(self, in_channels):
        super(Res1dUnit, self).__init__()
        temp_channel = int(in_channels/4)
        self.BR = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(), 
        )
        self.conv = nn.Sequential(
            NormConv1d(in_channels,temp_channel,1),
            NormConv1d(temp_channel,temp_channel,3),
            nn.Conv1d(temp_channel,in_channels,1),
        )

    def forward(self, x):
        b1 = x
        b2 = self.conv(self.BR(x))
        res = b1+b2
        return res

class Res1dProjectionUnit(nn.Module):
    '''
    模型开始前的预卷积
    '''
    def __init__(self, in_channels,temp_channels,out_channels,stride=2):
        super(Res1dProjectionUnit, self).__init__()

        self.branch1 = nn.Sequential(
            NormConv1d(in_channels,temp_channels,kernel_size=1,stride=stride,padding="valid"),
            NormConv1d(temp_channels,temp_channels,kernel_size=3),
            nn.Conv1d(temp_channels,out_channels,kernel_size=1)
        )
        self.branch2 = nn.Conv1d(in_channels,out_channels,kernel_size=1,stride=stride,padding="valid")


    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        res = b1+b2
        return res

class MaxInterUnit(nn.Module):
    def __init__(self, in_channels):
        super(MaxInterUnit, self).__init__()
        self.maxp = nn.MaxPool2d(3,2,padding=1)
        self.res_unit = clones(ResUnit(in_channels),2)
    def forward(self, x):
        x = self.maxp(x)
        for layer in self.res_unit:
            x = layer(x)
        x = F.interpolate(x,scale_factor= 2,mode = 'bilinear')
        return x

class AddMax(nn.Module):
    def __init__(self, in_channels,model):
        super(AddMax, self).__init__()
        self.max = nn.Sequential(nn.MaxPool2d(3,2,padding=1),ResUnit(in_channels))
        self.inter_res = ResUnit(in_channels)
        self.model = model
    def forward(self, x):
        max = self.max(x)
        cut = max
        b1 = self.model(max)
        b2 = b1+cut
        res = self.inter_res(b2)
        res = F.interpolate(res,scale_factor=2,mode='bilinear')
        return res
    
class AttentionModule(nn.Module):
    def __init__(self, in_channels,m=2,p=1,t=1):
        super(AttentionModule, self).__init__()
        self.pre_res = clones(ResUnit(in_channels),p)
        self.trunk = clones(ResUnit(in_channels),t)
        self.att_res = clones(ResUnit(in_channels),4)
        self.mask_trunk = MaxInterUnit(in_channels)
        if m>0:
            self.mask_trunk = AddMax(in_channels,self.mask_trunk)
            m=m-1
        self.aft_res = nn.Sequential(nn.BatchNorm2d(in_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels,in_channels,1,1,padding='same'),
                                    nn.BatchNorm2d(in_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels,in_channels,1,1,padding='same'),
                                    nn.Sigmoid(),
        )
        self.last_res = ResUnit(in_channels)
    def forward(self, x):
        for layer in self.pre_res:
            x = layer(x)
        trunk = x
        for layer in self.trunk:
            trunk = layer(trunk)
        mask = x
        mask = self.mask_trunk(mask)
        mask = self.aft_res(mask)
        dot = torch.mul(mask,trunk)
        add = torch.add(dot,trunk)
        res = self.last_res(add)
        return res

class ResAttentionModule(nn.Module):
    def __init__(self):
        super(ResAttentionModule, self).__init__()
        self.pre_conv = nn.Sequential(NormConv(64,64,7,2,padding=3),
                                    NormMaxPoll(64,3,1,1),
                                    ResProjectionUnit(64,64,128,1)
        )
        self.trunk1 = AttentionModule(128,0)
        self.projetion1 = ResProjectionUnit(128,128,128,2)
        self.trunk2 = AttentionModule(128,0)
        self.projetion2 = ResProjectionUnit(128,128,128,2)
        # self.trunk3 = AttentionModule(1024,0)
        # self.projetion3 = ResProjectionUnit(1024,1024,2048,2)
        self.avg = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
        )
        self.avg_trans = nn.Linear(128,128)
    def forward(self, x):
        x = self.pre_conv(x)
        x = self.trunk1(x)
        x = self.projetion1(x)
        x = self.trunk2(x)
        x = self.projetion2(x)
        # x = self.trunk3(x)
        # x = self.projetion3(x)
        avg = self.avg(x)
        avg_trans = self.avg_trans(avg.squeeze()).reshape((avg.shape[0],avg.shape[1],1,1))
        x = x/avg_trans
        return x
    


class before_mult_att(nn.Module):
    def __init__(self, length=208,out_length=64):
        super(before_mult_att, self).__init__()
        self.att_layer = nn.Linear(length,out_length)

    def forward(self, x):
        '''
        x (batch_size,channel,range,d+a+a)
        '''
        x_shape = x.shape
        # x变为 (expand_batch,channel,d+a+a)
        x = x.reshape((-1,x_shape[3]))
        # 把qkv矩阵变为 (expand_batch,d+a+a -> length,channel)
        output = self.att_layer(x)
        output=output.reshape((x_shape[0],x_shape[1],x_shape[2],64))
        return output


class AttentionResNet(nn.Module):
    def __init__(self):
        super(AttentionResNet, self).__init__()
        self.before_mult_att = before_mult_att(208,64)
        self.branch = ResAttentionModule()
        self.res_process = nn.Sequential(
            Res1dUnit(128),
            Res1dProjectionUnit(128,64,128),
            Res1dUnit(128),
            Res1dProjectionUnit(128,128,256),
            Res1dUnit(256),
            Res1dProjectionUnit(256,128,256),
            Res1dUnit(256),
            nn.Flatten(),
        )
        self.ske_predict = nn.Sequential(
            nn.Linear(768,320),
            nn.ReLU(),
            nn.Linear(320,288),
            nn.Sigmoid(),
        )
        self.peo_predict = nn.Sequential(
            nn.Linear(768,320),
            nn.ReLU(),
            nn.Linear(320,3),
            nn.Sigmoid(),
        )
    def forward(self, rd,re,ra):
        # rd,re,ra = x
        cat = torch.concat((rd,re,ra),dim=-1)
        feat = self.before_mult_att(cat)
        feat = torch.transpose(feat,1,3)
        feat = self.branch(feat)
        res = self.res_process(feat.squeeze())
        ske_res = self.ske_predict(res)
        peo_res = self.peo_predict(res)

        # res = torch.reshape((res),(-1,3,32,1))
        return torch.concat((peo_res,ske_res),axis=(-1))
    
def make_model(

):
    "Helper: Construct a model from hyperparameters."
    # c = copy.deepcopy
    # attn = MultiHeadedAttention(head, d_model)
    # ff = ConvFeedForward(d_model, d_ff, kernel_size=5, dropout=dropout)

    model = AttentionResNet()

    # # This was important from their code.
    # # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

if __name__ == '__main__':
    input1 = torch.randn(16,8,144,80)
    input2 = torch.randn(16,8,144,64)
    input3 = torch.randn(16,8,144,64)
    norm_conv1 = AttentionResNet()
    output1 = norm_conv1(input1,input2,input3)

    # from torchsummary import summary
    # device = torch.device("cuda:0" if torch.cuda.is_available()
    #                       else "cpu")
    # # model.to(device)
    # summary(norm_conv1, torch.stack(input1.shape[1:],input2.shape[1:],input3.shape[1:]))
    pass
