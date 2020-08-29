"""This module hosts all of the functions that are used for visualization"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import confusion_matrix
from matplotlib import patches
from mlxtend.plotting import plot_confusion_matrix

def show_image(path):
    """This function plots the image of interest"""
    
    img = mpimg.imread(path)
    imgplot = plt.imshow(img)
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
    

# Make visualizations
def visualize_training_results(results):
  """
  function for visualizing the loss and accuracy metrics.
  """
  history = results.history
  plt.figure()
  plt.plot(history['val_loss'])
  plt.plot(history['loss'])
  plt.legend(['val_loss', 'loss'])
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.show()
    
  plt.figure()
  plt.plot(history['val_accuracy'])
  plt.plot(history['accuracy'])
  plt.legend(['val_accuracy', 'accuracy'])
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.show()

    
def plot_cm(labels, predictions, classes):
    cm = confusion_matrix(y_true=labels, y_pred=predictions)
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                    show_absolute=True,
                                    show_normed=True,
                                    colorbar=True,
                                    class_names=classes)
    plt.show()
    