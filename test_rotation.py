
import cv2
import numpy as np
import random

def test_rotation():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    center_x, center_y = 32, 32
    angle = random.uniform(0, 360)  # Angle in degrees
    rotated_img = cv2.warpAffine(img, cv2.getRotationMatrix2D((center_x, center_y), angle, 1), (64, 64))
    cv2.imwrite('rotated_test.png', rotated_img)

test_rotation()
