"""This module is used to store the functions that turn the raw data into features"""

import cv2
import numpy as np
import os
from skimage.transform import rotate

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


def shift_image(image, label, shift):
    """
    This function takes in an image, and returns an image shifted to the left or to the right
    """
    shifted_image = [np.roll(image, shift), label]
    
    return shifted_image

