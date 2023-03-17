ENGG5104 – Image Processing and Computer Vision – Spring 2023
Assignment 2 — Image Recognition
LEI Haoyu 1155107869

Task 1: Implement VGGNet
We are required to implement VGGNet-A with 11 weight layers in model.py. The whole model structure is implemented in class vggnet,
and can be accessed in train.py.

Task 2: CrossEntropy Loss
We are required to implement cross entropy loss function in loss.py without directly using functions in nn.functional. Following
the guidance of specification, the loss function is implemented by just using nn.Softmax. 

However, since this Cifar-10 dataset is imbalanced, so the final version of loss function assigns different weights to each classes to make higher accuracy. This 
change was made during Task 4.

Task 3: Conventional Augmentations
We are required to implement several image augmentation methods in transforms.py without directly using functions in transform. Following
the steps in specification, padding, random crops and random horizontal flip are implemented respectively. In this sequence, an image is 
at first changed from (32, 32, 3) to (40, 40, 3) using padding = 4, then randomly cropped from (40, 40, 3) to (32, 32, 3), finally
horizontally flipped with probability = 0.5.

Task 4: CIFAR10 Image Recognition
We are required to further improve the Top-1 accuracy of Cifar-10 image recognition. At first, one of the image augmentation methods, Cutout, 
is implemented in transforms.py and used in train.py, then as specified in Task 2, the weights are added to loss calculation to help solving 
imbalanced dataset. After those two changes, the Top-1 accuracy has been successfully increased from 61% to 68%.