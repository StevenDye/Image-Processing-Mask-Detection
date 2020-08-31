"""This module is for making the dataset"""
import sys
import numpy as np
sys.path.insert(1, '../src/features/')
import build_features as bf


TRAIN_PATH = '../data/raw/train'
TEST_PATH = '../data/raw/test'
IMG_SIZE = 50
CATEGORIES = ['with_mask', 'without_mask']


"""Prepare Training Data"""
# Import and prepare data
train_data = bf.prepare_data(IMG_SIZE, CATEGORIES, TRAIN_PATH)
    
# Create flipped augmented data
for idx in range(len(train_data)):
    train_data.append(bf.flip_image(train_data[idx][0], train_data[idx][1]))

# Create right-shifted augmented data
for idx in range(len(train_data)):
    train_data.append(bf.shift_image(train_data[idx][0], train_data[idx][1], 1))

# Create left-shifted augmented data
for idx in range(len(train_data)):
    train_data.append(bf.shift_image(train_data[idx][0], train_data[idx][1], -2))

# Create up-shifted augmented data
for idx in range(len(train_data)):
    train_data.append(bf.shift_image(train_data[idx][0], train_data[idx][1], IMG_SIZE))

# Create down-shifted augmented data
for idx in range(len(train_data)):
    train_data.append(bf.shift_image(train_data[idx][0], train_data[idx][1], -2*IMG_SIZE))

# Shuffle data
np.random.shuffle(train_data)

# Split data into train and test sets
train_images = []
train_labels = []

for idx in range(len(train_data)):
    train_images.append(train_data[idx][0])
    train_labels.append(train_data[idx][1])

train_images = np.array(train_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
train_labels = np.asarray(train_labels).reshape((-1, 1))

# Scale data
train_images = train_images/255.0


"""Prepare Testing Data"""
# Import and prepare data
test_data = bf.prepare_data(IMG_SIZE, CATEGORIES, TEST_PATH)

test_images = []
test_labels = []

for idx in range(len(test_data)):
    test_images.append(test_data[idx][0])
    test_labels.append(test_data[idx][1])

test_images = np.array(test_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_labels = np.asarray(test_labels).reshape((-1, 1))

# Scale data
test_images = test_images/255.0
