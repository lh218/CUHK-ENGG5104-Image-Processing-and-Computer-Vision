import numpy as np
import cv2


def hpf_fourier(im, hpf_w=60):
    # TODO Write "High-pass Filter with Fourier Transform" function based on the illustration in specification.
    # Return transformed result image
    f_img = np.fft.fft2(im)
    f_shift = np.fft.fftshift(f_img)
    (width, height) = im.shape
    half_width, half_height = int(width/2), int(height/2)
    f_shift[half_width - int(hpf_w/2) : half_width + int(hpf_w/2) + 1, half_height - int(hpf_w/2) : half_height + int(hpf_w/2) + 1] = 0
    result = np.fft.ifft2(np.fft.ifftshift(f_shift)).real
    return result

if __name__ == '__main__':
    im = cv2.imread('./misc/lena_gray.bmp', 0)
    
    hpf_w = 60
    result = hpf_fourier(im, hpf_w)
    cv2.imwrite('./results/hpf_fourier.png', result)