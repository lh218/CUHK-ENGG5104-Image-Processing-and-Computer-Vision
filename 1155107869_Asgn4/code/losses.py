import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EPELoss(nn.Module):
    def __init__(self, args, div_flow = 0.05):
        super(EPELoss, self).__init__()
        self.div_flow = div_flow 
        self.loss_labels = ['EPE'],

    def forward(self, output, target):
        epevalue = 0
        target = self.div_flow * target
        assert output.shape == target.shape, (output.shape, target.shape)
        ''' Implement the EPE loss here'''
        output = output.permute((0, 2, 3, 1))
        target = target.permute((0, 2, 3, 1))
        epevalue = torch.norm(target - output, p=2, dim=1).mean()
        return [epevalue]


class MultiscaleLoss(nn.Module):
    def __init__(self, args):
        super(MultiscaleLoss, self).__init__()

        self.args = args
        self.div_flow = 0.05
        self.loss_labels = ['Multiscale'],
        ''' Implement the MultiScale loss here'''
        ''''''

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0
        target = self.div_flow * target
        for i, output_ in enumerate(output):
            target_ = F.interpolate(target, output_.shape[2:], mode='bilinear', align_corners=False)
            assert output_.shape == target_.shape, (output_.shape, target_.shape)
            ''' Implement the MultiScale loss here'''
            weights = [0.32, 0.16, 0.08]
            output_ = output_.permute((0, 2, 3, 1))
            target_ = target_.permute((0, 2, 3, 1))
            epevalue += weights[i] * torch.norm(target_ - output_, p=2, dim=1).mean()
            
        return [epevalue]
