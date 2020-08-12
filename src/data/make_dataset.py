"""This module is for making the dataset"""

import numpy as np
from skimage.transform import rotate

def flip_image(image):
    """This function takes in an image and returns an image flipped on its y-axis, and returns it as an numpy array"""
    
    flipped_image = np.fliplr(image)
    
    return flipped_image


def rotate_image(image):
    """This function takes in an image, and returns an image that is rotated 90 degrees anticlockwise"""
    
    rotated_image = rotate(image, 90)
    
    return rotated_image