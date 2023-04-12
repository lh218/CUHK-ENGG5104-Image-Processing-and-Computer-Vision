import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

class FlowNetOurs(nn.Module):
    def __init__(self, args, input_channels = 12, div_flow=20):
        super(FlowNetOurs, self).__init__()
        
        self.rgb_max = args.rgb_max
        self.div_flow = div_flow    # A coefficient to obtain small output value for easy training, ignore it

        '''Implement Codes here'''
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(514, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(514, 256, kernel_size=4, stride=2, padding=1)
        
        self.conv_pred1 = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv_pred2 = nn.Conv2d(514, 2, kernel_size=3, stride=1, padding=1)
        self.conv_pred3 = nn.Conv2d(514, 2, kernel_size=3, stride=1, padding=1)
        self.conv_pred4 = nn.Sequential(
            nn.Conv2d(256+128+2, 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=4)
        )
        
        self.flow_upsample1 = nn.Upsample(scale_factor=2)
        self.flow_upsample2 = nn.Upsample(scale_factor=2)
        self.flow_upsample3 = nn.Upsample(scale_factor=2)
        
    def forward(self, inputs):
        ## input normalization
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:, :, 0, :, :], x[:, :, 1, :, :]), dim=1)
        
        
        x = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        flow5 = self.conv_pred1(x5)
        x6 = torch.cat((x4, self.deconv1(x5), self.flow_upsample1(flow5)), dim=1)
        flow4 = self.conv_pred2(x6)
        x7 = torch.cat((x3, self.deconv2(x6), self.flow_upsample2(flow4)), dim=1)
        flow3 = self.conv_pred3(x7)
        x8 = torch.cat((x2, self.deconv3(x7), self.flow_upsample3(flow3)), dim=1)
        flow2 = self.conv_pred4(x8)

        if self.training:
            return flow2, flow3, flow4, flow5
        else:
            return flow2 * self.div_flow