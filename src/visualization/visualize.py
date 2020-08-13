"""This module hosts all of the functions that are used for visualization"""
import matplotlib.pyplot as plt

def show_image(image, title='Image', cmap_type='gray'):
    """This function plots the image of interest"""
    
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()