import numpy as np
import cv2
from matplotlib import pyplot as plt

def histogram_equalization(im):
    # TODO Write "histogram equalization" function based on the illustration in specification.
    # Return filtered result image
    im_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    im_gray = im_yuv[:,:,0]
    im_equ = cv2.equalizeHist(im_gray)
    result = np.hstack((im_gray, im_equ))
    return result

def local_histogram_equalization(im):
    # TODO Write "local histogram equalization" function based on the illustration in specification.
    # Return filtered result image
    im_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    im_gray = im_yuv[:,:,0]
    equ_local = cv2.createCLAHE(clipLimit=25)
    im_local = equ_local.apply(im_gray)
    result = np.hstack((im_gray, im_local))
    return result

if __name__ == '__main__':
    im = cv2.imread('./misc/Original_HistEqualization.jpeg')
    
    result_hist_equalization = histogram_equalization(im)
    result_local_hist_equalization = local_histogram_equalization(im)

    cv2.imwrite('./results/HistoEqualization.jpg', result_hist_equalization)
    cv2.imwrite('./results/LocalHistoEqualization.jpg', result_local_hist_equalization)