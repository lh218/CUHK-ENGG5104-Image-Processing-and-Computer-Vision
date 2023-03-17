import torch
import torch.nn as nn
import torch.nn.functional as F

class vggnet(nn.Module):
    # TODO: task 1
    def __init__(self, cfg, num_classes):
        super(vggnet, self).__init__()
        self.in_channels = 3
        self.num_classes = num_classes
        
        self.layer_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.layer_linear = nn.Sequential(
            nn.Linear(in_features=512*1*1, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            
            nn.Linear(in_features=4096, out_features=self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_conv(x)
        x = x.view(x.size(0), -1)
        x = self.layer_linear(x)
        return x