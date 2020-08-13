"""This module is for making the dataset"""

import numpy as np
from skimage import color, measure
from skimage.feature import canny, Cascade, corner_harris
from skimage.filters import threshold_otsu
from skimage.transform import resize, rotate

def find_contours(image):
    """This function finds the contours of an image"""
    
    image = np.asarray(image)
    image = color.rgb2gray(image)
    # Obtain threshhold value
    thresh = threshold_otsu(image) 
    # Apply thresholding
    threshold_image = image > thresh
    # Find countours at a constant value of 0.8
    countours = measure.find_contours(threshold_image, 0.8)
    
    return countours

def find_corners(image):
    """This function finds the corners in an image"""
    
    image = np.asarray(image)
    image = color.rgb2gray(image)
    # Apply Harris corner detector
    measure_image = corner_harris(image) 
    
    return measure_image


def find_edges(image):
    """This function finds the edges of an image using canny"""
    
    image = np.asarray(image)
    image = color.rgb2gray(image) 
    # Apply Canny detector with a sigma value (default=1)
    canny_edges = canny(image, sigma=0.5)
    
    return canny_edges
    

def flip_image(image):
    """This function takes in an image and returns an image flipped on its y-axis, and returns it as an numpy array"""
    
    flipped_image = np.fliplr(image)
    
    return flipped_image


def resize_image(image, width, height):
    """This function takes in an image and resizes it to the specified size"""
    
    image = np.asarray(image)
    resized_image = resize(image, (height, width), anti_aliasing=True)  # anti_aliasing smooths out pixilation
    
    return resized_image


def rotate_image(image):
    """This function takes in an image, and returns an image that is rotated 90 degrees anticlockwise"""
    
    image = np.asarray(image)
    rotated_image = rotate(image, 90)
    
    return rotated_image