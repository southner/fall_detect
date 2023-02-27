import torch

aa = [True,True,False,False]
bb = [True,False,True,False]

aa = torch.tensor(aa)
bb = torch.tensor(bb)

cc =(aa == 0) & (bb == 0)
dd =  torch.sum(cc)
pass