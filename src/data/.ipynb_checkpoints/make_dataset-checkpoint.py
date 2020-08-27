"""This module is for making the dataset"""
import numpy as np
import sys

sys.path.insert(1, '../src/features/')
import build_features as bf

from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  img_to_array)
from sklearn.model_selection import train_test_split

ROOT_PATH = '../data/raw/'
IMG_SIZE = 50
CATEGORIES = ['with_mask', 'without_mask']

# Import and prepare data
data = bf.prepare_data(IMG_SIZE, CATEGORIES, ROOT_PATH)

#datagen = ImageDataGenerator(rotation_range=45,
#                             width_shift_range=0.2,
#                            height_shift_range=0.2,
#                            sheer_range=0.2,
#                            zoom_range=0.2,
#                            horizontal_flip=True,
#                            fill_mode='reflect')

#for batch in datagen.flow_from_directory(directory=ROOT_PATH,
#                                         bat_size=15,
#                                         target_size=(IMG_SIZE, IMG_SIZE),
#                                         color_mode='grayscale',)

# Create flipped augmented data
#for idx in range(len(data)):
#    data.append(bf.flip_image(data[idx][0], data[idx][1]))

# Create rotated augmented data
#for idx in range(len(data)):
#    data.append(bf.rotate_image(data[idx][0], data[idx][1], 180))

#for idx in range(len(data)):
#    data.append(bf.rotate_image(data[idx][0], data[idx][1], 90))

# Shuffle data 
shuffled_data = np.random.shuffle(data)

# Split data into train and test sets
X = []
y = []

for idx in range(len(data)):
    X.append(data[idx][0])
    y.append(data[idx][1])

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.asarray(y).reshape((-1,1))

# Scale data
X = X/255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
