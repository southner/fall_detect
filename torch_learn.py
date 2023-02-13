import torch

x= torch.randn(10,10,256,128)

X = x.reshape((5,-1,256,128))