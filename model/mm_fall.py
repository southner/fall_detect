from torch import nn
import torch
if __name__=='__main__':
    from utils import clones
else:
    from .utils import clones
import torch.nn.functional as F

class MmFall(nn.Module):
    '''
    模型开始前的预卷积
    '''
    def __init__(self):
        super(MmFall, self).__init__()
        self.p1 = nn.Linear(96,160)
        self.p2 = nn.Linear(160,320)
        self.p3 = nn.Linear(320,160)
        self.p4 = nn.Linear(160,40)
        self.p5 = nn.Linear(40,10)
        self.p6 = nn.Linear(10,1)
        
    def forward(self, x):
        b1 = self.p1(x)
        b2 = self.p2(b1)
        b3 = self.p3(b2)
        b4 = self.p4(b3)
        b5 = self.p5(b4)
        b6 = self.p6(b5)
        res = F.sigmoid(b6)
        return res
    

def make_model(

):
    "Helper: Construct a model from hyperparameters."
    # c = copy.deepcopy
    # attn = MultiHeadedAttention(head, d_model)
    # ff = ConvFeedForward(d_model, d_ff, kernel_size=5, dropout=dropout)

    model = SkeFall()

    # # This was important from their code.
    # # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model