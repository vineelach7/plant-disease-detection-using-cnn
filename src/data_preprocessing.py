import cv2
import os
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = image / 255.0  # Normalize
    return image
