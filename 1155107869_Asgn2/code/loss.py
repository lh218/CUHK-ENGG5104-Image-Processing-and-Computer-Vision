import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    # TODO: task 2
    
    def __init__(self, weights):
        super(CrossEntropyLoss, self).__init__()
        self.weights = weights
        # TODO: implemente cross entropy loss for task2;
        # You cannot directly use any loss functions from torch.nn or torch.nn.functional, other modules are free to use.
        
    def forward(self, x, y, **kwargs):
        eps = 1e-12
        softmax = F.softmax(x, dim=1)
        log_softmax = -1 * torch.log(softmax+eps)
        weight = torch.zeros(log_softmax.shape[0]).cuda()
        for idx, y_idx in enumerate(y):
            weight[idx] = self.weights[y_idx]
        loss = (log_softmax[range(y.shape[0]), y] * weight).mean()
        return loss
        
