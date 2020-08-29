# Mask-Detection
The project focuses on creating a machine learning model that can determine if a person in a photo is wearing a mask or not.

Date: July 2020 - August 2020

Collaborators: Steven Dye

## Data
The data comes from two sources, the Kaggle dataset "Face Mask ~12K Images Dataset", which can be found here: https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset, and from Ashish Jangra, who has images collected on their Github here: https://github.com/balajisrinivas/Face-Mask-Detection/tree/master/dataset. Augmented images from were removed from the data as we will be making our own augmented images, and other images were removed to keep with_mask and without_mask categories balanced.

## Summary of files
- README.md
- data
  - raw
    - test
      - with_mask: 738 images
      - without_mask: 754 images
    - train
      - with_mask: 1735 images
      - without_mask: 1735 images
- models
  - 50x50-5-epochs-111040-images.h5
- notebooks
  - Technical_Notebook.ipynb
- requirements.txt
- src
  - data
    - make_dataset.py
  - features
    - build_features.py
  - models
    - build_model.py
  - visualizations
    - visualize.py
