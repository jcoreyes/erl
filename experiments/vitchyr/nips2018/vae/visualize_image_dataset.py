import sys
import numpy as np
import cv2

imgs = np.load(sys.argv[1])
for img in imgs:
    cv2.imshow('img', img.reshape(3, 84, 84).transpose())
    cv2.waitKey(1)
