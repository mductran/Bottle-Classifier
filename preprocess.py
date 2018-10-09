"""
preprocess images
"""

import cv2
import numpy as np
import Augmentor as aug

def preprocess(path):
    p = aug.Pipeline(path)
    p.greyscale(1.0)
    p.shear(0.4, 25, 25)
    p.flip_left_right(0.4)
    p.flip_top_bottom(0.4)
    p.rotate_random_90(0.4)

if __name__ == "__main__":
    path = input()
    preprocess(path)