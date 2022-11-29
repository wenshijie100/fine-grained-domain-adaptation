import torch
import torch.nn as nn
from loss_funcs import *

class PersonLoss(nn.Module):
    def __init__(self, loss_type, **kwargs):
        super(PersonLoss, self).__init__()
        #print("KK",kwargs)
        self.loss_type = loss_type
        if loss_type == "person":
            self.loss_func = PERSONLOSS
        else:
            print("WARNING: No valid transfer loss function is used.")
            self.loss_func = lambda x, y: 0 # return 0
    
    def forward(self, source, target, **kwargs):
        return self.loss_func(source, target, **kwargs)