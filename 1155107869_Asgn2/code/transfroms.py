import torch
import numpy as np
import PIL
import cv2
import random
import torchvision.transforms as transforms

# TODO: implementation transformations for task3;
# You cannot directly use them from pytorch, but you are free to use functions from cv2 and PIL
class Padding(object):
    def __init__(self, padding):
        self.padding = padding

    def __call__(self, img, **kwargs):
        k = self.padding
        img1 = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
        border = cv2.copyMakeBorder(img1, k, k, k, k, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        resized = cv2.resize(border, (40, 40))
        img = PIL.Image.fromarray(cv2.cvtColor(resized,cv2.COLOR_BGR2RGB))
        return img

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, **kwargs):
        img1 = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
        h, w = img1.shape[0], img1.shape[1]
        y1 = torch.randint(0, h - self.size + 1, size=(1,)).item()
        x1 = torch.randint(0, w - self.size + 1, size=(1,)).item()
        y2 = y1 + self.size
        x2 = x1 + self.size
        img = img1[x1:x2, y1:y2]
        img = PIL.Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        return img


class RandomFlip(object):
    def __init__(self,):
        pass
    
    def __call__(self, img, **kwargs):
        if random.random() < 0.5:
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        return img

class Cutout(object):
    def __init__(self, num, length):
        self.num = num
        self.length = length
    
    def __call__(self, img, **kwargs):
        h, w = img.size(1), img.size(2)
        mask_kernel = np.ones((h, w), np.float32)

        for n in range(self.num):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask_kernel[y1: y2, x1: x2] = 0.

        mask_kernel = torch.from_numpy(mask_kernel)
        mask_kernel = mask_kernel.expand_as(img)
        img = img * mask_kernel
        
        return img