ENGG5104 – Image Processing and Computer Vision – Spring 2023
Assignment 4 — Optical Flow Estimation
LEI Haoyu 1155107869

For this assignment, we are requiered to finish 4 methods of FlowNet, under those two constraints: FLOPS less than 2300M, 
and parameters less than 5M. The results of my models is shown:

--------------------------------------
Methods      FLOPs      Para      EPE
FlowNet-E    2173.9M    4.59M     6.11
FlowNet-ER   2193.3M    4.59M     6.08
FlowNet-ERM  2202.4M    4.61M     5.73
FlowNet-Ours 2259.7M    4.03M     5.50
--------------------------------------

Note:
1. FlowNet-Ours is based on https://arxiv.org/pdf/1709.02371.pdf, which adds more layers to the FlowNet-ERM and modify the 
new loss function respectively.

2. You should notice that the FLOPs of FlowNet-Ours are higher than the others, but the number of parameters are much less. 
The reason is that: for FlowNet-E, ER, ERM, the last conv layer is (256, 512) and (512, 512) which can achieve a higher performance. 
However for FlowNet-Ours, the last conv layer is changed to (256, 256) and (256, 512) to satisfy the constraints.