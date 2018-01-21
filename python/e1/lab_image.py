#Author:wqh
#encoding:utf-8

import numpy as np
import cv2

image = cv2.imread("H:\\img\\lena.jpg")
cv2.imshow("Original",image)
cv2.waitKey(0)

#lab空间
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("L*a*b*", lab)
cv2.waitKey(0)