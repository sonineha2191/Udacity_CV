#!/usr/bin/env python
import cv2
import numpy as np


image = cv2.imread('David_Hannay_0001.jpg')
cv2.imshow('images',image)
x, y = print (image.shape)
print (image.dtype)

Noise = np.random.random((x, y))
Noisy_image = image+Noise
cv2.imwrite('Noisy_image.jpg',Noisy_image)
