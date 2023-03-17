import numpy as np
import cv2
import math

def line_det(im):
    # TODO Write "line detection" function based on the illustration in specification.
    # Return detected result image
    im_gray = cv2.GaussianBlur(im,(3,3),0)
    im_canny = cv2.Canny(im_gray, 50, 150)
    line_image = np.copy(im) * 0
    lines = cv2.HoughLinesP(im_canny, 1, math.pi/180, 40, np.array([]), 70, 6)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),2)
    result = cv2.addWeighted(im, 1, line_image, 1, 0)
    return result

if __name__ == '__main__':
    im = cv2.imread('./misc/road.jpeg')
    
    result = line_det(im)
    cv2.imwrite('./results/line_det.png', result)