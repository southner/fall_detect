import scipy.io as scio
import os.path 
import numpy as np
import torch 
import torch.nn as nn

input1 = torch.ones([32,108,64,64])
input2 = torch.ones([32,108,64,120])
model = nn.Sequential(
    nn.Conv2d(108,108,[3,3],[2,1],padding=[1,1]),
    )
output1 = model(input1)
output2 = model(input2)
pass