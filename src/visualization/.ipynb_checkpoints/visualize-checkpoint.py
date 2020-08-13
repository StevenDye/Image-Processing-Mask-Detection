"""This module hosts all of the functions that are used for visualization"""

import matplotlib.pyplot as plt
from matplotlib import patches

def show_image(image, title='Image', cmap_type='gray'):
    """This function plots the image of interest"""
    
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()
    
    
def show_detected_face(result, detected, title='Face Image'):
    """This function creates a square around detected faces"""
    plt.imshow(result)
    img_desc = plt.gca()
    plt.set_cmap('gray')
    plt.title(title)
    plt.axis('off')
    
    for patch in detected:
        img_desc.add_patch(
            patches.Rectangle(
                (patch['c'], patch['r']),
                patch['width'],
                patch['height'],
                fill=False, color='r', linewidth=2)
        )
    plt.show()
    