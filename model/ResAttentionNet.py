from torch import nn
import torch
if __name__ == '__main__':
    from utils import clones
else:
    from .utils import clones
import torch.nn.functional as F


class NormConv(nn.Module):
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
    模型开始前的预卷积
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
    def __init__(self, in_channels, model):
        super(AddMax, self).__init__()
        self.max = nn.Sequential(nn.MaxPool2d(
            3, 2, padding=1), ResUnit(in_channels))
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


class mult_att(nn.Module):
    def __init__(self, channels=64, length=18, out_length=13):
        super(mult_att, self).__init__()
        self.q_layer = nn.Sequential(
            nn.Linear(length, out_length),
        )
        self.k_layer = nn.Sequential(
            nn.Linear(length, length),
        )
        self.v_layer = nn.Sequential(
            nn.Linear(length, length),
        )
        self.mult_att = nn.MultiheadAttention(channels, 2, batch_first=True)
        self.channels = channels
        self.length = length
        self.out_length = out_length

    def forward(self, x):
        # 把时间当做channel
        '''
        x (batch_size,channel,range,d+a+a)
        '''
        x_shape = x.shape
        # x变为 (expand_batch,channel,d+a+a)
        x = x.permute((0, 2, 1, 3)).reshape((-1, x_shape[1], x_shape[3]))
        # 把qkv矩阵变为 (expand_batch,d+a+a -> length,channel)
        q = self.q_layer(x).permute((0, 2, 1))
        k = self.k_layer(x).permute((0, 2, 1))
        v = self.v_layer(x).permute((0, 2, 1))
        output, _ = self.mult_att(q, k, v)
        output = output.reshape(
            (x_shape[0], x_shape[2], self.out_length, x_shape[1]))
        # 当前维度为 (batch_size,range,new(d+a+a),channel)
        res = output.permute((0, 3, 1, 2))

        return res


class ResAttentionModule(nn.Module):
    def __init__(self):
        super(ResAttentionModule, self).__init__()
        self.pre_conv = nn.Sequential(NormConv(10, 10, 7, 2, padding=3),
                                      NormMaxPoll(10, 3, 1, 1),
                                      ResProjectionUnit(10, 10, 32, 1)
                                      )
        self.trunk1 = AttentionModule(32, 0)
        self.projetion1 = ResProjectionUnit(32, 48, 64, 2)
        self.trunk2 = AttentionModule(64, 0)
        self.projetion2 = ResProjectionUnit(64, 96, 128, 2)
        # self.trunk3 = AttentionModule(1024,0)
        # self.projetion3 = ResProjectionUnit(1024,1024,2048,2)
        self.avg = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                 )
        self.avg_trans = nn.Linear(128, 128)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.trunk1(x)
        x = self.projetion1(x)
        x = self.trunk2(x)
        x = self.projetion2(x)
        # x = self.trunk3(x)
        # x = self.projetion3(x)

        # avg = self.avg(x)
        # avg_trans = self.avg_trans(avg.squeeze()).reshape((avg.shape[0],avg.shape[1],1,1))
        # x = x/avg_trans
        return x





# class TransformerBlock(nn.Module):
#     # Vision Transformer https://arxiv.org/abs/2010.11929
#     def __init__(self, c1, c2, num_heads, num_layers):
#         super().__init__()
#         self.conv = None
#         if c1 != c2:
#             self.conv = nn.Conv(c1, c2)
#         self.linear = nn.Linear(c2, c2)  # learnable position embedding
#         self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
#         self.c2 = c2

#     def forward(self, x):
#         if self.conv is not None:
#             x = self.conv(x)
#         b, _, w, h = x.shape
#         p = x.flatten(2).permute(2, 0, 1)
#         return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    '''
    c : embedding_chan
    未做layernorm
    '''
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads, batch_first=True)
        # self.norm = nn.LayerNorm()
    def forward(self, x):
        batch,chan,rd,ra = x.size()
        q = self.q(x.permute(0, 2, 3, 1).reshape(-1, rd, chan))
        k = self.k(x.permute(0, 2, 3, 1).reshape(-1, rd, chan))
        v = self.v(x.permute(0, 2, 3, 1).reshape(-1, rd, chan))

        x = self.ma(q,k,v)[0].reshape(-1, rd, ra, chan).permute(0,3,1,2) + x
        return x

class AttFusion(nn.Module):
    '''
    Fusion of heat map RA/RV used SelfAttention

    rv : heatmap range-velocity [batch,channel,d_r , d_v]
    ra : heatmap range-azimuth  [batch,channel,d_r , d_a]  

    未做 layernorm
    '''
    def __init__(self, d_r, d_a, d_v, chan):
        super().__init__()
        self.d_r = d_r
        self.d_a = d_a
        self.d_v = d_v
        self.chan = chan
        self.q = nn.Linear(d_v, d_v, bias=False)
        self.k = nn.Linear(d_a, d_a, bias=False)
        self.v = nn.Linear(d_a, d_a, bias=False)
        num_heads = 2
        self.ra_att = TransformerLayer(chan,num_heads)
        self.rv_att = TransformerLayer(chan,num_heads)
        self.ma = nn.MultiheadAttention(embed_dim=self.chan, num_heads=num_heads,batch_first=True)

    def forward(self, ra, rv):
        assert ra.shape[:2] == rv.shape[:2]
        # 使用linear 从rv的chan转换为embedding_chan
        ra = self.ra_att(ra)
        rv = self.ra_att(rv)
        q = self.q(rv.permute(0, 2, 3, 1).reshape(-1, self.d_v, self.chan))
        k = self.k(ra.permute(0, 2, 3, 1).reshape(-1, self.d_a, self.chan))
        v = self.v(ra.permute(0, 2, 3, 1).reshape(-1, self.d_a, self.chan))
        
        temp_rv,weight = self.ma(q, k, v)
        trans_rv = temp_rv.reshape(-1, self.d_r, self.d_v, self.chan).permute(0,3,1,2)
        return trans_rv + rv
    
class ConvFusion(nn.Module):
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
            ResUnit(self.chan),
            nn.ReLU()
        )
        self.norm_2 = nn.Sequential(
            ResUnit(self.chan),
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
        self.ra_to_rv = ConvFusion(d_r, d_a, d_v, chan)
        self.rv_to_ra = ConvFusion(d_r, d_v, d_a, chan)
        # or
        # self.ra_to_rv = AttFusion(d_r, d_a, d_v, chan)
        # self.rv_to_ra = AttFusion(d_r, d_v, d_a, chan)
        
        self.rv_conv = NormConv(chan, chan*2)
        self.ra_conv = NormConv(chan, chan*2)
        self.ra_max_pool = nn.AdaptiveAvgPool2d([int(d_r/2), int(d_a/2)])
        self.rv_max_pool = nn.AdaptiveAvgPool2d([int(d_r/2), int(d_v/2)])

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
        self.ra_to_rv = ConvFusion(d_r, d_a, d_v, chan)
        self.rv_to_ra = ConvFusion(d_r, d_v, d_a, chan)
        # or
        # self.ra_to_rv = AttFusion(d_r, d_a, d_v, chan)
        # self.rv_to_ra = AttFusion(d_r, d_v, d_a, chan)
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


# class UAttentionNet(nn.Module):
#     '''
#     一个UNet下采样module
#     '''

#     def __init__(self, d_r, d_a, d_v, chan):
#         super(UAttentionNet, self).__init__()

#         self.rv_att_1 = AttentionModule(chan, m=2)
#         self.ra_att_1 = AttentionModule(chan, m=2)
#         self.cross_att_1 = BioFusion(d_r, d_a, d_v, chan)
#         self.rv_att_2 = AttentionModule(chan*2, m=1)
#         self.ra_att_2 = AttentionModule(chan*2, m=1)
#         self.cross_att_2 = BioFusion(
#             int(d_r/2), int(d_a/2), int(d_v/2), chan*2)
#         self.rv_att_3 = AttentionModule(chan*4, m=0)

#     def forward(self, rv, ra):

#         rv_att_1 = self.rv_att_1(rv)
#         ra_att_1 = self.ra_att_1(ra)
#         rv_cross_att_1, ra_cross_att_1 = self.cross_att_1(rv_att_1, ra_att_1)
#         rv_att_2 = self.rv_att_2(rv_cross_att_1)
#         ra_att_2 = self.ra_att_2(ra_cross_att_1)
#         rv_cross_att_2, ra_cross_att_2 = self.cross_att_2(rv_att_2, ra_att_2)
#         rv_att_3 = self.rv_att_3(rv_cross_att_2)
#         return rv_att_3

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

        return ra_cross_att_3, F.interpolate(self.final_up(up_cross_ra_att_1), scale_factor=2, mode='bilinear')
    

class ResAttentionNet(nn.Module):
    def __init__(self, d_r, d_a, d_v, chan):
        super(ResAttentionNet, self).__init__()
        self.rv_pre = nn.Sequential(
            NormConv(chan, 16),
            ResUnit(16),
            NormConv(16, 32),
            ResUnit(32)
        )
        self.ra_pre = nn.Sequential(
            NormConv(chan, 16),
            ResUnit(16),
            NormConv(16, 32),
            ResUnit(32)
        )
        self.sj_net = UAttentionNet(d_r, d_a, d_v, 32)
        self.paf_net = UAttentionNet(d_r, d_a, d_v, 32)
        
        self.sj_predict_branch =nn.Sequential(
            ResUnit(64),
            ResProjectionUnit(64,48,64,1),
            nn.AdaptiveAvgPool2d([46,82]),
            ResUnit(64),
            ResProjectionUnit(64,48,32,1),
            ResUnit(32),
            ResProjectionUnit(32,30,26,1),
            ResUnit(26),
            ResUnit(26),
        )
        self.paf_predict_branch =nn.Sequential(
            ResUnit(64),
            ResProjectionUnit(64,48,64,1),
            nn.AdaptiveAvgPool2d([46,82]),
            ResUnit(64),
            ResProjectionUnit(64,64,52,1),
            ResUnit(52),
            ResUnit(52),
        )
        self.heatmap_branch = nn.Sequential(
            nn.Sigmoid()
        )

        self.fall_res_process = nn.Sequential(
            ResUnit(256),
            CutConv(256, 128, [3, 3], [1, 1], [1, 2]),
            ResUnit(128),
            CutConv(128, 64, [3, 2], [1, 1], [1, 0]),
            ResUnit(64),
            CutConv(64, 32, [3, 2], [1, 1], [1, 1]),
            ResUnit(32),
            CutConv(32, 12, [3, 2], [1, 1], [1, 0]),
            ResUnit(12),
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, rd, ra):
        # rd,re,ra = x
        rv_att_0 = self.rv_pre(rd)
        ra_att_0 = self.ra_pre(ra)
        fall_att_1,sj_att = self.sj_net(rv_att_0, ra_att_0)
        fall_att_2,paf_att = self.paf_net(rv_att_0, ra_att_0)

        res = self.fall_res_process(fall_att_1+fall_att_2)
        fall_res = torch.permute(res, [0, 2, 3, 1])
        fall_res[:, :, :, [0, 2, 4, 5, 7, 9]] = F.sigmoid(fall_res[:, :, :, [0, 2, 4, 5, 7, 9]])
        fall_res[:, :, :, [1, 3, 6, 8]] = F.relu(fall_res[:, :, :, [1, 3, 6, 8]])
        fall_res[:, :, :, [10, 11]] = F.softmax(fall_res[:, :, :, [10, 11]], dim=3)
        
        sj = self.sj_predict_branch(sj_att)
        paf = self.paf_predict_branch(paf_att)
        heatmap = self.sigmoid(torch.concat([sj,paf],dim=1))
        return fall_res,heatmap


def make_model():
    "Helper: Construct a model from hyperparameters."
    # c = copy.deepcopy
    # attn = MultiHeadedAttention(head, d_model)
    # ff = ConvFeedForward(d_model, d_ff, kernel_size=5, dropout=dropout)

    model = ResAttentionNet(64, 64, 120, 10)

    # # This was important from their code.
    # # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


if __name__ == '__main__':
    rd = torch.randn(16, 10, 64, 120)
    ra = torch.randn(16, 10, 64, 64)
    re = torch.randn(16, 10, 64, 64)
    norm_conv1 = ResAttentionNet(64, 64, 120, 10)
    output1 = norm_conv1(rd, ra)
    print("11")

    # model = AttFusion(64,64,120,10)
    # model(ra,rd)

    # model = TransformerLayer(10,2)
    # model(ra)
    pass
    # from torchsummary import summary
    # device = torch.device("cuda:0" if torch.cuda.is_available()
    #                       else "cpu")
    # # model.to(device)
    # summary(norm_conv1, torch.stack(input1.shape[1:],input2.shape[1:],input3.shape[1:]))
    pass
