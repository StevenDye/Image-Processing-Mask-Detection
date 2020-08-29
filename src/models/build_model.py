"""This module is to build the model"""
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.layers import (Dense,
                                     Dropout,
                                     Flatten,
                                     Conv2D,
                                     MaxPooling2D,
                                     Activation)
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

INPUT_SHAPE = (50, 50, 1)

model = Sequential()
model.add(Conv2D(32, (3, 3),
                 input_shape=INPUT_SHAPE,
                 kernel_regularizer=regularizers.l2(1e-5),
                 bias_regularizer=regularizers.l2(1e-5),
                 activity_regularizer=regularizers.l2(1e-5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(1e-5),
                 bias_regularizer=regularizers.l2(1e-5),
                 activity_regularizer=regularizers.l2(1e-5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, kernel_regularizer=regularizers.l2(1e-5),
                bias_regularizer=regularizers.l2(1e-5),
                activity_regularizer=regularizers.l2(1e-5)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])