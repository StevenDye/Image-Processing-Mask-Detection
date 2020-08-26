"""This module is used to store the functions that turn the raw data into features"""

import cv2
import numpy as np
import os
from skimage import color, measure
from skimage.feature import canny, Cascade, corner_harris
from skimage.filters import threshold_otsu
from skimage.transform import resize, rotate

def prepare_data(img_size, categories, root_path):
  """
  This function pulls the images from the data folder, gives them a label
  depending on what folder they are pulled from, grayscales the image, and
  resizes the image the specified size.
  """
  data = []
  for category in categories:
    img_path = os.path.join(root_path, category)
    class_num = categories.index(category)  # get the classification  (0 or a 1)
    for img in os.listdir(img_path):
      try:
        img_array = cv2.imread(os.path.join(img_path, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (img_size, img_size))  # resize to normalize data size
        data.append([new_array, class_num])  # add this to our data
      except Exception as e:
        pass

  return data


def flip_image(image, label):
    """
    This function takes in an image and returns an image flipped on its y-axis,
    and returns it in a list with a label.
    """
    flipped_image = [np.fliplr(image), label]
    
    return flipped_image



def rotate_image(image, label, angle):
    """
    This function takes in an image, and returns an image that is an angle anticlockwise
    """
    rotated_image = [rotate(image, angle), label]
    
    return rotated_image












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
    

def resize_image(image, width, height):
    """This function takes in an image and resizes it to the specified size"""
    
    image = np.asarray(image)
    resized_image = resize(image, (height, width), anti_aliasing=True)  # anti_aliasing smooths out pixilation
    
    return resized_image
