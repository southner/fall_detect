from torch import nn
import torch
if __name__ == '__main__':
    from utils import clones
else:
    from .utils import clones
import torch.nn.functional as F


class NormConv(nn.Module):
    'conv2d+Batch+Relu'

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding="same"):
        super(NormConv, self).__init__()

        self.m = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
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
    'Pool2d+BatchNorm+RELU'

    def __init__(self, featuires, kernel_size=1, stride=1, padding=0):
        super(NormMaxPoll, self).__init__()

        self.m = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size,
                         stride=stride, padding=padding),
            nn.BatchNorm2d(featuires),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.m(x)


class ResProjectionUnit(nn.Module):
    '''
    根据stride参数改变尺寸的ResLayer
    '''

    def __init__(self, in_channels, temp_channels, out_channels, stride=2):
        super(ResProjectionUnit, self).__init__()

        self.branch1 = nn.Sequential(
            NormConv(in_channels, temp_channels, kernel_size=1,
                     stride=stride, padding="valid"),
            NormConv(temp_channels, temp_channels, kernel_size=3),
            nn.Conv2d(temp_channels, out_channels, kernel_size=1)
        )
        self.branch2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride, padding="valid")

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        res = b1+b2
        return res


class ResUnit(nn.Module):
    '''
    不改变尺寸的ResLayer
    '''

    def __init__(self, in_channels):
        super(ResUnit, self).__init__()
        temp_channel = int(in_channels/4)
        self.BR = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            NormConv(in_channels, temp_channel, 1),
            NormConv(temp_channel, temp_channel, 3),
            nn.Conv2d(temp_channel, in_channels, 1),
        )

    def forward(self, x):
        b1 = x
        b2 = self.conv(self.BR(x))
        res = b1+b2
        return res


class CutConv(nn.Module):
    '自定义kernel_size、stride、padding的relu'

    def __init__(self, in_channels, out_channels, kernel_size=[2, 3], stride=[1, 2], padding=0):
        super(CutConv, self).__init__()

        self.m = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            # BatchNorm2d是针对通道的归一化，具体对于(N,C,W,H)的输入，针对每一个C，计算一个均值和方差
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.m(x)


class MaxInterUnit(nn.Module):
    '一次maxPool+interpolate'

    def __init__(self, in_channels):
        super(MaxInterUnit, self).__init__()
        self.maxp = nn.MaxPool2d(3, 2, padding=1)
        self.res_unit = clones(ResUnit(in_channels), 2)

    def forward(self, x):
        x = self.maxp(x)
        for layer in self.res_unit:
            x = layer(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        return x


class AddMax(nn.Module):
    '在前后加上一层maxPool+interpolate 一层cut'

    def __init__(self, in_channels, model):
        super(AddMax, self).__init__()
        self.max = nn.Sequential(nn.MaxPool2d(3, 2, padding=1),
                                 ResUnit(in_channels))
        self.inter_res = ResUnit(in_channels)
        self.model = model

    def forward(self, x):
        max = self.max(x)
        cut = max
        b1 = self.model(max)
        b2 = b1+cut
        res = self.inter_res(b2)
        res = F.interpolate(res, scale_factor=2, mode='bilinear')
        return res


class AttentionModule(nn.Module):
    '''
    利用高层特征监督低层特征
    m:内部会进行m+1次maxPool
    '''

    def __init__(self, in_channels, m=2, p=1, t=1):
        super(AttentionModule, self).__init__()
        self.pre_res = clones(ResUnit(in_channels), p)
        self.trunk = clones(ResUnit(in_channels), t)
        self.att_res = clones(ResUnit(in_channels), 4)
        self.mask_trunk = MaxInterUnit(in_channels)
        if m > 0:
            self.mask_trunk = AddMax(in_channels, self.mask_trunk)
            m = m-1
        self.aft_res = nn.Sequential(nn.BatchNorm2d(in_channels),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels, in_channels,
                                               1, 1, padding='same'),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels, in_channels,
                                               1, 1, padding='same'),
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
        dot = torch.mul(mask, trunk)
        add = torch.add(dot, trunk)
        res = self.last_res(add)
        return res


class SingleFusion(nn.Module):
    '''
    Fusion of heat map RA/RV used ConV2d

    rv : heatmap range-velocity [batch,channel,d_r , d_v]
    ra : heatmap range-azimuth  [batch,channel,d_r , d_a]  

    convert ra to rv

    INPUT
    d-v: dimensions of velocity
    d-a: dimensions of azimuth
    d-r: dimensions of range
    chan: dimensions of channel
    from d-a to d-v
    '''

    def __init__(self, d_r, d_a, d_v, chan):
        super().__init__()

        self.range_d = d_r
        self.velo_d = d_v
        self.azi_d = d_a
        self.chan = chan
        self.conv_1 = nn.Conv1d(d_a, d_v, kernel_size=1, stride=1)
        self.conv_2 = nn.Conv1d(d_a, d_v, kernel_size=1, stride=1)

        self.norm_1 = nn.Sequential(
            nn.BatchNorm2d(self.chan),
            nn.ReLU()
        )
        self.norm_2 = nn.Sequential(
            nn.BatchNorm2d(self.chan),
            nn.ReLU()
        )

        self.pro_v1 = nn.Sequential(
            NormConv(self.chan, self.chan),
        )
        self.pro_v2 = nn.Sequential(
            NormConv(self.chan, self.chan),
        )
        self.relu = nn.ReLU()

    def forward(self, ra, rv):
        assert ra.shape[:2] == rv.shape[:2]

        batch, _, __, ___ = ra.shape
        '''
        ra : heatmap range-azimuth  [batch,channel,d_r , d_a]
        to [batch,d_r , d_a,channel]
        to [batch*d_r , d_a,channel]
        to [batch*d_r , d_v,channel]
        to [batch,d_r , d_v,channel]
        to [batch,channel , d_r,d_v]
        '''
        map_1 = self.conv_1(ra.permute(0, 2, 3, 1).reshape(-1, self.azi_d, self.chan)).reshape(batch, self.range_d, -1,
                                                                                               self.chan).permute(0,
                                                                                                                  3,
                                                                                                                  1,
                                                                                                                  2)
        map_2 = self.conv_2(ra.permute(0, 2, 3, 1).reshape(-1, self.azi_d, self.chan)).reshape(batch, self.range_d, -1,
                                                                                               self.chan).permute(0,
                                                                                                                  3,
                                                                                                                  1,
                                                                                                                  2)
        weight = self.norm_1(map_1)
        bias = self.norm_2(map_2)
        rv = self.pro_v1(rv)
        cut = rv
        rv = self.pro_v2(rv)
        out = self.relu(rv * weight + bias)+cut
        return out


class BioFusion(nn.Module):
    '''
    进行双向Fusion
    输入：
    rv : heatmap range-velocity [batch,channel,d_r , d_v]
    ra : heatmap range-azimuth  [batch,channel,d_r , d_a] 
    输出：
    rv : heatmap range-velocity [batch,channel,d_r , d_v]
    ra : heatmap range-azimuth  [batch,channel,d_r , d_a] 
    '''

    def __init__(self, d_r, d_a, d_v, chan):
        super().__init__()
        self.ra_to_rv = SingleFusion(d_r, d_a, d_v, chan)
        self.rv_to_ra = SingleFusion(d_r, d_v, d_a, chan)
        self.rv_conv = NormConv(chan, chan*2)
        self.ra_conv = NormConv(chan, chan*2)
        self.ra_max_pool = nn.AdaptiveAvgPool2d([int(d_r/2),int(d_a/2)])
        self.rv_max_pool = nn.AdaptiveAvgPool2d([int(d_r/2),int(d_v/2)])

    def forward(self, rv, ra):
        assert ra.shape[:2] == rv.shape[:2]
        f_rv = self.rv_conv(self.ra_to_rv(ra, rv))
        f_ra = self.ra_conv(self.rv_to_ra(rv, ra))
        return self.rv_max_pool(f_rv), self.ra_max_pool(f_ra)


class BioUpFusion(nn.Module):
    '''
    进行双向Fusion
    输入：
    rv : heatmap range-velocity [batch,channel,d_r , d_v]
    ra : heatmap range-azimuth  [batch,channel,d_r , d_a] 
    输出：
    rv : heatmap range-velocity [batch,channel,d_r , d_v]
    ra : heatmap range-azimuth  [batch,channel,d_r , d_a] 
    '''

    def __init__(self, d_r, d_a, d_v, chan):
        super().__init__()
        self.ra_to_rv = SingleFusion(d_r, d_a, d_v, chan)
        self.rv_to_ra = SingleFusion(d_r, d_v, d_a, chan)
        self.rv_conv = nn.Sequential(
            NormConv(chan, int(chan/2)),
            NormConv(int(chan/2), int(chan/4)),
            )
        self.ra_conv = nn.Sequential(
            NormConv(chan, int(chan/2)),
            NormConv(int(chan/2), int(chan/4)),
            )

    def forward(self, rv, ra):
        assert ra.shape[:2] == rv.shape[:2]
        f_rv = self.rv_conv(self.ra_to_rv(ra, rv))
        f_ra = self.ra_conv(self.rv_to_ra(rv, ra))
        return F.interpolate(f_rv, scale_factor=2, mode='bilinear'), F.interpolate(f_ra, scale_factor=2, mode='bilinear')
        

class UAttentionNet(nn.Module):
    '''
    一个UNet下采样module
    '''

    def __init__(self, d_r, d_a, d_v, chan):
        super(UAttentionNet, self).__init__()
        
        self.rv_att_1 = AttentionModule(chan, m=2)
        self.ra_att_1 = AttentionModule(chan, m=2)
        self.cross_att_1 = BioFusion(d_r, d_a, d_v, chan)
        self.rv_att_2 = AttentionModule(chan*2, m=1)
        self.ra_att_2 = AttentionModule(chan*2, m=1)
        self.cross_att_2 = BioFusion(int(d_r/2), int(d_a/2), int(d_v/2), chan*2)
        self.rv_att_3 = AttentionModule(chan*4, m=0)
        self.ra_att_3 = AttentionModule(chan*4, m=0)
        self.cross_att_3 = BioFusion(int(d_r/4), int(d_a/4), int(d_v/4), chan*4)
        
        self.up_rv_att_3 = ResUnit(chan*8)
        self.up_ra_att_3 = ResUnit(chan*8)
        self.up_cross_att_3 = BioUpFusion(int(d_r/8), int(d_a/8), int(d_v/8), chan*16)
        self.up_rv_att_2 = AttentionModule(chan*4, m=0)
        self.up_ra_att_2 = AttentionModule(chan*4, m=0)
        self.up_cross_att_2 = BioUpFusion(int(d_r/4), int(d_a/4), int(d_v/4), chan*8)
        self.up_rv_att_1 = AttentionModule(chan*2, m=1)
        self.up_ra_att_1 = AttentionModule(chan*2, m=1)
        self.up_cross_att_1 = BioUpFusion(int(d_r/2), int(d_a/2), int(d_v/2), chan*4)
        self.final_up = nn.Sequential(
            NormConv(chan,chan*2),
            ResUnit(chan*2),
        )
        
        
    def forward(self, rv, ra):
        
        rv_att_1 = self.rv_att_1(rv)
        ra_att_1 = self.ra_att_1(ra)
        rv_cross_att_1, ra_cross_att_1 = self.cross_att_1(rv_att_1, ra_att_1)
        rv_att_2 = self.rv_att_2(rv_cross_att_1)
        ra_att_2 = self.ra_att_2(ra_cross_att_1)
        rv_cross_att_2, ra_cross_att_2 = self.cross_att_2(rv_att_2, ra_att_2)
        rv_att_3 = self.rv_att_3(rv_cross_att_2)
        ra_att_3 = self.ra_att_3(ra_cross_att_2)
        rv_cross_att_3, ra_cross_att_3 = self.cross_att_3(rv_att_3, ra_att_3)
        
        up_rv_att_3 = self.up_rv_att_3(rv_cross_att_3)
        up_ra_att_3 = self.up_ra_att_3(ra_cross_att_3)
        up_cross_rv_att_3,up_cross_ra_att_3 = self.up_cross_att_3(torch.concat([up_rv_att_3,rv_cross_att_3],dim=1),torch.concat([up_ra_att_3,ra_cross_att_3],dim=1))
        up_rv_att_2 = self.up_rv_att_2(up_cross_rv_att_3)
        up_ra_att_2 = self.up_ra_att_2(up_cross_ra_att_3)
        up_cross_rv_att_2,up_cross_ra_att_2 = self.up_cross_att_2(torch.concat([up_rv_att_2,rv_cross_att_2],dim=1),torch.concat([up_ra_att_2,ra_cross_att_2],dim=1))
        up_rv_att_1 = self.up_rv_att_1(up_cross_rv_att_2)
        up_ra_att_1 = self.up_ra_att_1(up_cross_ra_att_2)
        up_cross_rv_att_1,up_cross_ra_att_1 = self.up_cross_att_1(torch.concat([up_rv_att_1,rv_cross_att_1],dim=1),torch.concat([up_ra_att_1,ra_cross_att_1],dim=1))

        return F.interpolate(self.final_up(up_cross_ra_att_1), scale_factor=2, mode='bilinear')

class HeatMapPredictNet(nn.Module):
    def __init__(self, d_r, d_a, d_v, chan):
        super(HeatMapPredictNet, self).__init__()
        self.rv_pre = nn.Sequential(
            NormConv(chan, 32),
            ResUnit(32),
            NormConv(32, 64),
            ResUnit(64)
        )
        self.ra_pre = nn.Sequential(
            NormConv(3, 32),
            ResUnit(32),
            NormConv(32, 64),
            ResUnit(64)
        )
        self.sj_u_net = UAttentionNet(d_r, d_a, d_v, 64)
        self.paf_u_net = UAttentionNet(d_r, d_a, d_v, 64)
        self.sj_predict_branch =nn.Sequential(
            ResUnit(128),
            ResProjectionUnit(128,96,64,1),
            nn.AdaptiveAvgPool2d([46,82]),
            ResUnit(64),
            ResProjectionUnit(64,48,32,1),
            ResUnit(32),
            ResProjectionUnit(32,30,26,1),
            ResUnit(26),
            ResUnit(26),
        )
        self.paf_predict_branch =nn.Sequential(
            ResUnit(128),
            ResProjectionUnit(128,96,64,1),
            nn.AdaptiveAvgPool2d([46,82]),
            ResUnit(64),
            ResProjectionUnit(64,64,52,1),
            ResUnit(52),
            ResUnit(52),
        )
        
    def forward(self, rv, ra):
        rv_att_0 = self.rv_pre(rv)
        ra_att_0 = self.ra_pre(ra)
        sj_att = self.sj_u_net(rv_att_0,ra_att_0)
        paf_att = self.paf_u_net(rv_att_0,ra_att_0)
        sj = self.sj_predict_branch(sj_att)
        paf = self.paf_predict_branch(paf_att)
        return sj,paf

def make_model():
    "Helper: Construct a model from hyperparameters."
    model = HeatMapPredictNet(64, 64, 120, 3)

    # # This was important from their code.
    # # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


if __name__ == '__main__':
    rv = torch.randn(16, 3, 64, 120)
    ra = torch.randn(16, 3, 64, 64)
    re = torch.randn(16, 3, 64, 64)
    norm_conv1 = HeatMapPredictNet(64, 64, 120, 3)
    sj,paf = norm_conv1(rv, ra)
    print(sj.shape)
    print(paf.shape)
