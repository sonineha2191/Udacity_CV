#!/usr/bin/env python
# img = imread('dolphin.png'); grey scale
# imshow(img);
# disp(size(img));
# disp(class(img));
# cropped = img(101:103,201:203)
# imgc = imread('dolphin.png'); colour image
# imgc_r = imgc(:,:,1)
# disp (size(imgc_r))
#  plot(img_red(150, :));
# imgc_g = img(:,:,2);
# imshow (imgc_g)
# plot(imgc_g(150, :));
# adding two images


import cv2
import numpy as np

def im_levels(image, no_levels):
    x, y = cv2.image.shape
    step_size = int(255/(no_levels-1))
    levels = list(range(0,255,step_size))
    for levels:
        for i in range (x):
        for j in range (y):
            if image[i,j] <  0:
                image[i,j] = 0
            if image[i,j] > no_levels:
                image[i,j] = no_levels
    return

image = cv2.imread('dolphin.jpg', 0)
cv2.imshow('images',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
x, y = image.shape
print (image.dtype)

Sigma = 50
# Noise = np.random.randn((x, y)) * Sigma
Noise = np.zeros((x ,y), np.uint8)
Noise = cv2.randn(Noise, 0, Sigma)
cv2.imshow('Noise',Noise)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Noise = Noise.astype(uint8)
# Noise = cv2.cvtColor(Noise, cv2.COLOR_BGR2GRAY)


cv2.imwrite('Noise.jpg',Noise)
Noisy_image = image + Noise
cv2.imshow('Noisy_image',Noisy_image)
cv2.waitKey(0)

cv2.imwrite('Noisy_image.jpg',Noisy_image)

cv2.destroyAllWindows()

