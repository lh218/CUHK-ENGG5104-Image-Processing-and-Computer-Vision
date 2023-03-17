import numpy as np
import cv2
import math

def image_t(im, scale=0.5, rot=45, trans=(50,-50)):
    # TODO Write "image affine transformation" function based on the illustration in specification.
    # Return transformed result image
    rad = rot * math.pi / 180
    width, height, rgb = im.shape
    center = (width/2, height/2)
    
    rot_mat = cv2.getRotationMatrix2D(center, rot, scale)
    rot_img = cv2.warpAffine(im, rot_mat, (width, height))
    
    pts1 = np.float32([[0, 0], [width, 0], [-width, height]])
    pts2 = np.float32([[pts1[0][0] + trans[0], pts1[0][1] + trans[1]], [pts1[1][0] + trans[0], pts1[1][1] + trans[1]], [pts1[2][0] + trans[0], pts1[2][1] + trans[1]]])
    
    tran_mat = cv2.getAffineTransform(pts1, pts2)
    result = cv2.warpAffine(rot_img, tran_mat, (height, width))
    
    return result


if __name__ == '__main__':
    im = cv2.imread('./misc/pearl.jpeg')
    
    scale  = 0.5
    rot    = 45
    trans  = (50, -50)
    result = image_t(im, scale, rot, trans)
    cv2.imwrite('./results/affine_result.png', result)